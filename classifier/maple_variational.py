"""
Author: Saurav Jha
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
import numpy as np

from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import dataset.incremental_dataloader

from .utils import build_cosine_scheduler, freeze_parameters
import pdb
import time
from .evaluator import Evaluator

from torch.distributions.normal import Normal 
from torch.distributions.kl import kl_divergence

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model, proj, ctx_vectors, prompt_prefix, compound_prompt_projections, compound_prompts_text):
        super().__init__()
        self.args = args
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype

        n_cls = len(classnames)
        self.dtype = dtype
        clip_imsize = clip_model.visual.input_resolution
        n_ctx = 2
        ctx_init = "a photo of a" 
        cfg_imsize = 224 
        self.compound_prompts_depth = 9

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.ctx = ctx_vectors
        self.proj = proj 
        self.compound_prompt_projections = compound_prompt_projections
        self.compound_prompts_text = compound_prompts_text

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda(device=self.args.default_gpu)).type(self.dtype)
        
        self.register_buffer( 'token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer( 'token_suffix', embedding[:, 1+n_ctx:,:]) # CLS, EOS, [n_cls, -1, ctx_dim]

        self.n_cls = n_cls 
        self.n_ctx = n_ctx 
        self.ctx_dim = ctx_dim
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model, proj, ctx_vectors, prompt_prefix, compound_prompt_projections, compound_prompts_text,
                  vga=None,  mu_adapters=None, sigma_adapters=None, task_tokens=None, 
                 task_to_cls_num=None, prompt_templates=None, previous_components=None,
                 task_to_distribution=None, mu_global_adapter=None, sigma_global_adapter=None,
                 mu_adapter_deter=None, global_vga=None):
        super().__init__()
        self.n_class = len(class_names)
        self.args = args
        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        # if torch.cuda.device_count() > 1:
        #     self.text_encoder = nn.DataParallel(self.text_encoder, device_ids=args.gpus)
        self.prompt_learner = MultiModalPromptLearner(args, class_names, clip_model,  proj, ctx_vectors, 
                                                        prompt_prefix, compound_prompt_projections, compound_prompts_text)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        dtype = clip_model.dtype
        self.ctx = ctx_vectors
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        normal_clip_model, _ = load(args.ckpt_path, device=f"cuda:{args.default_gpu}", 
                               )
        normal_clip_model.eval()
        self.pretrained_text_encoder = normal_clip_model.encode_text

        self.current_class_names = class_names
        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        previous_ctx = None
        if previous_components is not None:
            self.unpack_prev_components(previous_components)

        self.vga = vga 
        self.vga_global = global_vga

        self.mu_adapters = mu_adapters
        self.sigma_adapters = sigma_adapters
        self.mu_global_adapter = mu_global_adapter
        self.sigma_global_adapter = sigma_global_adapter
        self.mu_adapter_deter = mu_adapter_deter

        self.forward_times = self.args.forward_times

        self.task_tokens = task_tokens
        self.task_to_cls_num = task_to_cls_num
        self.prompt_templates = prompt_templates
        self.prior_text_features()
        self.class_to_task_mapping = {} # for faster indexing to get task ids
        self.init_new_heads()

    def init_new_heads(self):
        def get_new_task_embed(var=False):
            if var:
                new_class_embeds = self.frozen_text_features_individual.var(1)
            else:
                new_class_embeds = self.frozen_text_features_individual.mean(1)
            layer_embeds = new_class_embeds.t() @ new_class_embeds
            # layer_embeds = layer_embeds / layer_embeds.norm()
            layer_embeds = layer_embeds / layer_embeds.shape[0]   
            return layer_embeds  
    
        def init_with_task_embed(module, var=False):
            layer_embeds = get_new_task_embed(var=var)
            for m in module.fc.children():
                if isinstance(m, torch.nn.Linear):
                    m.weight.copy_(layer_embeds)
        with torch.no_grad():
            init_with_task_embed(self.mu_adapters[-1])
            init_with_task_embed(self.sigma_adapters[-1], var=True)

    def unpack_prev_components(self, previous_components):
        previous_mu, previous_sigma, previous_vga, previous_mu_global_adapter, previous_sigma_global_adapter  = previous_components
        self.previous_mu_adapters = previous_mu
        self.previous_sigma_adapters = previous_sigma
        self.previous_vga = previous_vga
        self.previous_mu_global_adapter, self.previous_sigma_global_adapter = previous_mu_global_adapter, previous_sigma_global_adapter

    @torch.no_grad()
    def prior_text_features(self):
        prompts = [[temp.format(c.replace("_", " ")) for temp in self.prompt_templates] for c in self.current_class_names]
        text_features_, text_features_per_prompt = [], []
        for per_cls_prompts in prompts:
            per_cls_prompt_embs = tokenize(per_cls_prompts).cuda(device=self.args.default_gpu)
            text_features = self.pretrained_text_encoder(per_cls_prompt_embs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_per_prompt.append(text_features)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
            text_features_.append(text_features)
        self.frozen_text_features = torch.stack(text_features_, dim=0)
        self.frozen_text_features_individual = torch.stack(text_features_per_prompt, dim=0)

    def get_variational_adapter_features(self, x, i=None, distill=False, global_adapter=False):
        if global_adapter:
            mu_adapter = self.previous_mu_global_adapter if distill else self.mu_global_adapter
            sigma_adapter = self.previous_sigma_global_adapter if distill else self.sigma_global_adapter
        else:
            mu_adapter = self.previous_mu_adapters[i] if distill else self.mu_adapters[i] 
            sigma_adapter = self.previous_sigma_adapters[i] if distill else self.sigma_adapters[i]
        mu = mu_adapter(x)
        sigma = sigma_adapter(x)
        dist = Normal(mu, sigma)
        return dist

    def get_attention_mask(self, attn_shape, nb_task_tokens, original_query_num):
        """Mask so that task tokens don't interact together.

        Given two task tokens (t1, t2) and three patch tokens (p1, p2, p3), the
        attention matrix is:

        t1-t1 t1-t2 t1-p1 t1-p2 t1-p3
        t2-t1 t2-t2 t2-p1 t2-p2 t2-p3

        So that the mask (True values are deleted) should be:

        False True False False False
        True False False False False
        """
        # self.task_to_cls_num[0] = 2
        # self.task_to_cls_num[1] = 2
        # self.task_to_cls_num[2] = 2
        # nb_task_tokens = 3
        # original_query_num = 6
        # attn_shape = (9, 9)
        mask = torch.zeros(attn_shape, dtype=torch.bool).cuda(device=self.args.default_gpu)
        if self.args.expandable_tokens:
            for i in range(nb_task_tokens):
                mask[original_query_num+i, original_query_num:original_query_num+i] = True
                mask[original_query_num+i, original_query_num+i+1:original_query_num+nb_task_tokens] = True

        start_cls_idx, end_cls_idx = 0, 0 
        for i in range(nb_task_tokens):
            start_cls_idx = end_cls_idx
            end_cls_idx += self.task_to_cls_num[i]
            curr_class_indices = np.arange(start_cls_idx, end_cls_idx)
            for cls in curr_class_indices:
                mask[cls][:start_cls_idx] = True 
                mask[cls][end_cls_idx:] = True
                if self.args.expandable_tokens:
                    mask[cls][original_query_num+i] = False
            if self.args.expandable_tokens:
                mask[original_query_num+i, :start_cls_idx] = True 
                mask[original_query_num+i, end_cls_idx:original_query_num] = True
        return mask

    @staticmethod
    def get_contrastive_matrix(text_feats, image_feats, logit_scale=None):
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        if logit_scale is not None:
            image_feats = image_feats.clone() * logit_scale
        contrastive_matrix = image_feats @ text_feats.t() # 16 x 16 matrix
        return contrastive_matrix

        def get_prior_dist(self, image_features=None, text_features=None, batch_labels=None, task_num=None, task_specific_labels=None, task_token=None, use_np_prior=False, global_adapter=False, tgt_mask=None):
            return Normal(torch.zeros_like(text_features), torch.ones_like(text_features))
        

    def forward(self, image, labels=None, test=False, finetuning=False, return_mean=True, for_prior=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        n_class = self.n_class
        prev_cls_num = self.n_class - self.task_to_cls_num[self.args.sess]
        if test:
            with torch.no_grad():
                text_features = self.text_encoder(self.prompts, tokenized_prompts, self.deep_compound_prompts_text)
                image_features = self.image_encoder(image.type(self.dtype), self.shared_ctx, self.deep_compound_prompts_vision) 
                image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)

                context = image_features_normed.clone() # torch.cat([image_features.unsqueeze(0), self.task_token_two[-1]], 1)
                n_query = text_features.shape[0]
                query = text_features.clone().unsqueeze(0)
                if self.args.expandable_tokens:
                    query = torch.cat([query] + [token for token in self.task_tokens], 1)                
                attn_mask = self.get_attention_mask((query.shape[1], query.shape[1]), self.args.sess+1, text_features.shape[0])
                if self.args.use_vga:
                    vga_features = self.vga(query, context.unsqueeze(0), tgt_mask=attn_mask).squeeze(0)
                logits =[]
                samplewise_text_feats = []
                start_cls_idx, end_cls_idx = 0, 0
                for i in range(self.args.sess+1):
                    start_cls_idx = end_cls_idx
                    end_cls_idx += self.task_to_cls_num[i]
                    text_features_relevant = text_features[start_cls_idx:end_cls_idx].clone()
                    text_features_ = text_features_relevant
                    if self.args.use_vga:
                        text_features_ = text_features_ + vga_features[start_cls_idx:end_cls_idx] 
                    qdist = self.get_variational_adapter_features(text_features_, i if self.args.expandable_adapter else 0)            
                    rsamples = qdist.rsample([self.forward_times])
                    text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1)
                    text_features_ = rsamples + text_features_ 
                    logits_ = logit_scale * image_features_normed @ text_features_.permute(0, 2, 1) 
                  
                    logits.append(logits_)
                logits = torch.cat(logits, -1)
                logits = logits.detach()
            if return_mean:
                return logits.mean(0), (None, None)
            else:
                return logits, (None,None)
        else:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
            image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = text_features.view(n_class, -1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits =[]
            kl_losses = []
            prior_matching_losses = []
            start_cls_idx, end_cls_idx = 0, 0
            context = image_features_normed.clone() 
            n_query = text_features.shape[0]
            query = text_features.clone().unsqueeze(0)
            attn_mask = self.get_attention_mask((query.shape[1], query.shape[1]), self.args.sess+1, text_features.shape[0])
            if self.args.use_vga:
                vga_features_all = self.vga(query, context.unsqueeze(0), tgt_mask=attn_mask).squeeze(0)

            per_sample_text_feats = []
            taskwise_means = []

            for i in range(self.args.sess+1):
                
                start_cls_idx = end_cls_idx
                end_cls_idx += self.task_to_cls_num[i]
                if start_cls_idx not in self.class_to_task_mapping:
                    # update class to task mapping for faster indexing of task id based on class label id
                    self.class_to_task_mapping.update(dict(zip(np.arange(start_cls_idx, end_cls_idx), [i] * (end_cls_idx - start_cls_idx))))

                text_features_relevant = text_features.clone()[start_cls_idx:end_cls_idx]
                if self.args.use_vga:
                    vga_features = vga_features_all[start_cls_idx:end_cls_idx]
                    if self.args.expandable_tokens:
                        vga_features = vga_features + vga_features_all[n_query+i]
                    text_features_ = text_features_relevant + vga_features
                else:
                    text_features_ = text_features_relevant

                qdist = self.get_variational_adapter_features(text_features_, i if self.args.expandable_adapter else 0)            
                rsamples = qdist.rsample([self.forward_times])
                text_features_ = text_features_.unsqueeze(0).expand(self.forward_times, -1, -1)
            
                text_features_ = rsamples + text_features_ 
                taskwise_means.append(rsamples.mean(0))
                if self.args.lasp  and self.args.beta > 0 and (finetuning or (not finetuning and  self.args.sess == i)):
                    prior_text_features = self.frozen_text_features_individual.clone()[start_cls_idx:end_cls_idx]
                    sims = torch.stack([prior_text_features @ rsamples[r].t() for r in range(rsamples.shape[0])], 0)
                    sims = sims.mean(2).mean(0)
                    kl_losses.append(F.cross_entropy(sims,  torch.arange(sims.size(0)).cuda(device=self.args.default_gpu)) * self.args.beta)
                    if self.args.use_det_path:
                        sims_det = prior_text_features @ deterministic_features.t() 
                        sims_det = sims_det.mean(1)
                        kl_losses.append(F.cross_entropy(sims_det,  torch.arange(sims_det.size(0)).cuda(device=self.args.default_gpu)) * self.args.beta)
                logits_ = (logit_scale * image_features_normed @ text_features_.permute(0, 2, 1)) 
                if finetuning or (not finetuning and self.args.sess == i):
                    pdist = self.get_prior_dist(context, text_features_relevant, labels, i, 
                                            None, 
                                            self.task_tokens[i] if self.args.expandable_tokens else None,
                                            use_np_prior=self.args.use_np_prior if not finetuning else False
                                            )
                    prior_matching_losses.append(kl_divergence(qdist, pdist).mean(0).sum() * 0.001)    
                
                logits.append(logits_)
                if (self.args.get_interclass_dist and self.args.sess == 9 and finetuning) or (self.args.get_adapter_distances and self.args.sess > 0):
                    with torch.no_grad():                        
                        per_sample_text_feats.append(rsamples.clone().detach().mean(0))
            
           

            # text_features_all = torch.cat(text_features_all, 0)
            logits = torch.cat(logits, -1)
            kl_loss = sum(kl_losses)  if len(kl_losses) else 0.
            prior_matching_loss = sum(prior_matching_losses) 
            # prior_matching_loss = prior_matching_loss * 0.01 #if not finetuning else prior_matching_loss * 0.1 
            
            avg_cos_distance = None              
            return logits, (kl_loss, prior_matching_loss, avg_cos_distance)

    def get_prior_dist(self, image_features=None, text_features=None, batch_labels=None, task_num=None, task_specific_labels=None, task_token=None, use_np_prior=False, global_adapter=False):
        return Normal(torch.zeros_like(text_features), torch.ones_like(text_features))
        

    @torch.no_grad()
    def set_classifier(self):
        tokenized_prompts = self.tokenized_prompts
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        self.shared_ctx = shared_ctx
        self.deep_compound_prompts_vision = deep_compound_prompts_vision
        self.deep_compound_prompts_text = deep_compound_prompts_text
        self.prompts = prompts

    @property #变成属性
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype #return int/float

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=False, layer_num=1):
        super().__init__()

        self.fc = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))
        self.sigma = sigma
        # init_weights(self.fc)

    def forward(self, x):
        if self.sigma:
            return F.softplus(self.fc(x)) * 0.999 + 0.001
        else:
            return self.fc(x)
        
class MaPLe_var(Evaluator):
    def __init__(self, args, n_ctx=16, use_float32=False, use_grad_checkpoint=False):
        super().__init__(args)
        self.args = args
        clip_model, _ = load(args.ckpt_path, device=f"cuda:{args.default_gpu}", 
                                design_details={"trainer": 'MaPLe',
                                        "vision_depth": 0,
                                        "language_depth": 0, "vision_ctx": 0,
                                        "language_ctx": 0,
                                        "maple_length": 2})
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        n_ctx = 2
        ctx_init = "a photo of a" 
        cfg_imsize = 224 
        self.compound_prompts_depth = 9

        self.proj = nn.Linear(ctx_dim, 768).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.cuda(device=self.args.default_gpu)).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers
        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512,  dtype=self.clip_model.dtype)).cuda(device=self.args.default_gpu)
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        self.n_ctx = n_ctx # n_ctx 输入词数
        self.lr = args.lr#*args.train_batch/20
        self.wd = args.wd # wd ??
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.current_class_names = []

        self.wd = args.wd # wd ??
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.current_class_names = []
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=ctx_dim, nhead=ctx_dim//64, activation='gelu', batch_first=True).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)
        self.vga = torch.nn.TransformerDecoder(decoder_layer, 1)

        self.get_variational_adapters(ctx_dim)
        self.vga_global = None 
        self.task_to_cls_num = {}
        self.task_to_distribution = {}

        # for distillation
        self.previous_ctx = None
        self.previous_mu_adapters, self.previous_mu_global_adapter = None, None
        self.previous_sigma_adapters, self.previous_sigma_global_adapter = None, None
        self.previous_task_tokens = None
        self.previous_vga = None

    @staticmethod
    def get_div_logits(outputs, nb_old_classes, nb_new_classes):
        outputs_div = outputs[:, :, nb_old_classes:nb_old_classes+nb_new_classes] 
        outputs_old = outputs[:, :, :nb_old_classes].max(-1)[0].unsqueeze(-1)
        outputs_div = torch.cat([outputs_old, outputs_div],  -1)
        return outputs_div
    
    def get_div_loss(self, outputs_div, div_targets):
        nb_old_classes = sum([self.task_to_cls_num[t_num] for t_num in range(self.args.sess)])

        mask_old_cls = div_targets < nb_old_classes
        mask_new_cls = ~mask_old_cls
        div_targets[mask_old_cls] = 0
        div_targets[mask_new_cls] -= nb_old_classes - 1
        aux_loss = F.cross_entropy(outputs_div.view(-1, outputs_div.shape[-1]), 
                                       div_targets) 
        return aux_loss

    def get_variational_adapters(self, ctx_dim, global_adapter=False):
        if not global_adapter:
            self.mu_adapters = nn.ModuleList([Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)])
            self.sigma_adapters = nn.ModuleList([Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)])
            self.mu_adapter_deter = None
            if self.args.use_det_path:
                self.mu_adapter_deter = nn.ModuleList([Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)])
        else:
            self.mu_global_adapter = Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)
            self.sigma_global_adapter = Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)

    def post_training(self, finalize=False):
        self.model.eval()
        self.model.set_classifier()
        if self.args.distill and finalize:
            self.preserve_copy_for_distillation()

    def fit(self, data):
        self.task_to_cls_num[self.args.sess] = len(data['class_names'])
        self.current_class_names += data['class_names']
        print(f"Classes: {self.current_class_names}")
        train_loader = data['train_loader']

        if len(train_loader.dataset)< self.train_batch:
            real_img_bsz = len(train_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch

        per_epoch_steps = len(train_loader)

        self.init_model(class_names=self.current_class_names, per_epoch_steps=per_epoch_steps, prompt_templates=data['prompt_templates'])

        inter_adapter_distances = []
        # self.model.eval()
        self.model.vga.train()
        if self.args.sess >= 0:
            for epoch in tqdm(range(self.epochs)):
                for idx, (x, y, index) in tqdm(enumerate(train_loader), total=len(train_loader), desc = 'Training'):

                    cur_iter_idx = epoch*per_epoch_steps+idx
                    self.cur_iter_idx = cur_iter_idx
                    self.scheduler.step(cur_iter_idx)

                    output, (kl_loss, prior_matching_loss, inter_adapter_distance) = self.model(x.cuda(device=self.args.default_gpu), y)
                    y = y.cuda(device=self.args.default_gpu)
                    loss = 0.
                    # pdb.set_trace()
                    if self.args.variational:
                        targets = y.unsqueeze(0).expand(output.shape[0], -1).contiguous().view(-1)
                        # if self.args.sess > 0:
                        #     loss = loss + self.get_div_loss(output.clone(), targets.clone()) * 0.01
                        output = output.view(-1, output.shape[-1])
                    else:
                        targets = y 
                    loss = loss + F.cross_entropy(output, targets) + kl_loss + prior_matching_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if inter_adapter_distance is not None and (epoch == self.epochs-1):
                        inter_adapter_distances.append(inter_adapter_distance)

            if self.args.sess > 0 and self.args.expandable_tokens:
                self.epoch_log()
            if len(inter_adapter_distances):
                print(f"Average inter-adapter distance: {np.mean(inter_adapter_distance)}")
        # if self.args.sess == 9 and self.args.get_interclass_dist:
        #     with torch.no_grad():
        #         self.compute_class_centroids()

        # pdb.set_trace()
            # print(self.model.prompt_learner.ctx)
            # print(self.model.image_encoder.layer1[0].conv1.weight[0])
        self.model.eval()
        self.model.vga.train()
        return self.model
 
    def finetuning(self, data):
        self.unfreeze_for_finetuning()
        self.cur_iter_idx = 0
        memory_loader = data['memory_loader']
        if len(memory_loader.dataset)< self.train_batch:
            real_img_bsz = len(memory_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch
            
        per_epoch_steps=len(memory_loader)
        inter_adapter_distances = []
        self.build_optimizer(per_epoch_steps=per_epoch_steps, lr=self.lr/10., warmup=False, finetune=True)
        self.model.vga.eval()
        for epoch in tqdm(range(self.args.finetune_epochs)):
            for idx, (x, y, index) in tqdm(enumerate(memory_loader), total=len(memory_loader), desc = 'Finetuning'):

                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)

                output, (kl_loss, prior_matching_loss, inter_adapter_distance) = self.model(x.cuda(device=self.args.default_gpu), y, finetuning=True)
                # pdb.set_trace()
                y = y.cuda(device=self.args.default_gpu)
                # pdb.set_trace()
                loss = 0.
                if self.args.variational:
                    targets = y.unsqueeze(0).expand(output.shape[0], -1).contiguous().view(-1)
                    output = output.view(-1, output.shape[-1])
                else:
                    targets = y 
                loss = loss + F.cross_entropy(output, targets) + kl_loss + prior_matching_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if inter_adapter_distance is not None and (epoch == self.epochs-1):
                        inter_adapter_distances.append(inter_adapter_distance)
        if self.args.sess == 9 and self.args.get_interclass_dist:
            with torch.no_grad():
                self.compute_class_centroids()
        if len(inter_adapter_distances):
                print(f"Average inter-adapter distance: {np.mean(inter_adapter_distance)}")

        if self.args.sess > 0 and self.args.expandable_tokens:
            self.epoch_log()
        
    @torch.no_grad()
    def preserve_copy_for_distillation(self):
        pass 

    def expand_adapter(self):
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        dtype = self.clip_model.dtype
        new_mu = Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(dtype)
        new_sigma = Adapter(ctx_dim, ctx_dim, sigma=True).cuda(device=self.args.default_gpu).type(dtype)
        self.mu_adapters.append(new_mu)
        self.sigma_adapters.append(new_sigma)
        self.mu_adapters[:-1].eval()
        self.sigma_adapters[:-1].eval()
        freeze_parameters(self.mu_adapters[:-1], requires_grad=False)
        freeze_parameters(self.sigma_adapters[:-1], requires_grad=False)
        freeze_parameters(self.mu_adapters[-1], requires_grad=True)
        freeze_parameters(self.sigma_adapters[-1], requires_grad=True)

    def unfreeze_for_finetuning(self, requires_grad=True):
        freeze_parameters(self.vga, requires_grad=False)
        freeze_parameters(self.mu_adapters[:-1], requires_grad=requires_grad)
        freeze_parameters(self.sigma_adapters[:-1], requires_grad=requires_grad)
        if requires_grad:
            self.mu_adapters[:-1].train()
            self.sigma_adapters[:-1].train()

    def init_model(self, class_names, per_epoch_steps,  prompt_templates=None):

        if self.args.sess > 0:
            freeze_parameters(self.vga, requires_grad=True)
            if self.args.expandable_tokens:
                self.expand_task_token_list()
            if self.args.expandable_adapter:
                self.expand_adapter()
            if self.args.expandable_prompt:
                self.expand_prompts()
        self.n_class = len(class_names)
        clip_model = deepcopy(self.clip_model)
        print(f"Number of prompt vectors: {len(self.ctx)}")

        prev_model_components = (
                                 self.previous_mu_adapters, self.previous_sigma_adapters, 
                                 self.previous_vga, 
                                 self.previous_mu_global_adapter, self.previous_sigma_global_adapter,
                                 )
        self.model = CLIP(self.args, class_names, clip_model, self.proj, self.ctx, self.prompt_prefix, 
                                        self.compound_prompt_projections, self.compound_prompts_text,
                                        vga = self.vga,  mu_adapters=self.mu_adapters, sigma_adapters=self.sigma_adapters,
                                        task_to_cls_num = self.task_to_cls_num,
                                        prompt_templates=prompt_templates, previous_components=prev_model_components,
                                        task_to_distribution=self.task_to_distribution,
                                        mu_global_adapter=self.mu_global_adapter if self.args.hierarchical else None, 
                                        sigma_global_adapter=self.sigma_global_adapter if self.args.hierarchical else None,
                                        mu_adapter_deter=self.mu_adapter_deter, global_vga=self.vga_global)
        
        self.model.eval()
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=self.args.gpus)
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True 
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True
        self.build_optimizer(per_epoch_steps, lr=self.lr, warmup=True)

    def build_optimizer(self, per_epoch_steps, lr, warmup=False, finetune=False):
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if "ctx" not in name and "vga" not in name and "task_token" not in name and "adapter" not in name:
                param.requires_grad_(False)
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                
        
        # double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"\nParameters to be updated: {sorted(enabled)}\n")

        param_dict = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]
        total_step=self.epochs*per_epoch_steps if not finetune else self.args.finetune_epochs*per_epoch_steps
        self.optimizer = torch.optim.SGD(param_dict, lr=lr, weight_decay=self.wd)
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=lr,
            total_step=total_step)

    @torch.no_grad()
    def inference(self,image, label, num_test, test_class):
        self.model.eval()
        logits, feats = self.model(image, label, test=True)
        return logits.float(), feats


    


