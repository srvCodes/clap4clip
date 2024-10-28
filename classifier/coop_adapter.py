import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal 
from torch.distributions.kl import kl_divergence

from tqdm import tqdm
from copy import deepcopy
import numpy as np

from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import dataset.incremental_dataloader

from .utils import build_cosine_scheduler
import pdb
import time

class PromptLearner(nn.Module):
    def __init__(self, args, class_names, clip_model, n_ctx=16, prompt_pos=2):
        super().__init__()
        self.args = args
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype

        n_cls = len(class_names)
        self.dtype = dtype
        ctx_vectors = torch.empty(1, n_ctx, ctx_dim, dtype=self.dtype).cuda(device=self.args.default_gpu)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix =' '.join(['x'] * n_ctx)
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]

        classnames = [name.replace('_', ' ') for name in class_names]
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]

        self.prompt_pos = prompt_pos

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda(device=self.args.default_gpu)).type(self.dtype)
        self.register_buffer( 'token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer( 'token_suffix', embedding[:, 1+n_ctx:,:]) # CLS, EOS, [n_cls, -1, ctx_dim]

        self.n_cls = n_cls 
        self.n_ctx = n_ctx 
        self.ctx_dim = ctx_dim

    def forward(self):

        ctx=self.ctx

        tokenized_prompts = self.tokenized_prompts.view(self.n_cls,-1)

        n_cls = self.n_cls

        if self.prompt_pos == 2:
            prefix = self.token_prefix.unsqueeze(1)
            suffix = self.token_suffix.unsqueeze(1)
            ctx = ctx.unsqueeze(0).repeat(n_cls, 1, 1, 1)
            prompts = torch.cat([prefix, ctx, suffix],dim=2)
        elif self.prompt_pos == 1:
            prompts =[]
            half_n_ctx = self.n_ctx // 2
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1,:name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i_half1 = ctx[:,:half_n_ctx, :].unsqueeze(0)
                ctx_i_half2 = ctx[:, half_n_ctx:,:].unsqueeze(0)
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.prompt_pos == 0:
            prompts =[]
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1,:,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1, :name_len,:].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        prompts = prompts.view(n_cls, -1, self.ctx_dim)

        return prompts, tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype) #position_embeding可训练
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # @ and
        return x

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.sigma = sigma
        self.init_weights(self.fc)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        if self.sigma:
            return F.softplus(self.fc(x))
        else:
            return self.fc(x)
    

class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model, n_ctx=16, prompt_templates=None):
        super().__init__()
        self.current_class_names = class_names
        self.n_class = len(class_names)
        self.args = args
        self.pretrained_text_encoder = clip_model.encode_text
        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder, device_ids=args.gpus)

        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.prompt_learner = PromptLearner(args, class_names, clip_model, n_ctx=n_ctx)

        # image encoder
        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        self.mu_adapter = Adapter(ctx_dim, ctx_dim).cuda(device=self.args.default_gpu).type(dtype)
        self.prompt_templates = prompt_templates

    def get_adapter_features(self, x):
        mu = self.mu_adapter(x)
        return mu
    
    @torch.no_grad()
    def prior_text_features(self):
        prompts = [[temp.format(c.replace("_", " ")) for temp in self.prompt_templates] for c in self.current_class_names]
        text_features_ = []
        for per_cls_prompts in prompts:
            per_cls_prompt_embs = tokenize(per_cls_prompts).cuda(device=self.args.default_gpu)
            text_features = self.pretrained_text_encoder(per_cls_prompt_embs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
            text_features_.append(text_features)
        text_features_ = torch.stack(text_features_, dim=0)
        return text_features_

    def forward(self, image, test=False):

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()
        n_class = self.n_class

        # image_features = image_features.unsqueeze(0).expand(self.forward_times, -1, -1)
        if test:
            text_features = self.text_features
            logit_scale = self.logit_scale.exp()
            mu = self.get_adapter_features(text_features)
            text_features = mu #text_features.unsqueeze(0).expand(self.forward_times, -1, -1) + zs * 0.1
            logits = logit_scale * image_features @ text_features.t()

            return logits

        else:
            text_prompt, tokenized_prompts = self.prompt_learner()
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            
            # prior_text_features = self.prior_text_features()
            mu = self.get_adapter_features(text_features)
            # p_dist = self.get_adapter_features(prior_text_features)
            ratio = 0.2
            text_features = mu #* ratio + text_features * (1-ratio)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(n_class, -1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            kl_loss = 0 #kl_divergence(Normal(torch.zeros_like(p_dist.loc), p_dist.scale), Normal(torch.zeros_like(q_dist.loc), q_dist.scale)).sum(-1).mean() * 0.3 # self.args.prior_matching_loss
           
            # logits_context = logit_scale * image_features @ prior_text_features.t()
            # logits_context = logits_context.unsqueeze(0).expand(self.forward_times, -1, -1)
            scl_loss = 0 #nn.KLDivLoss(reduction='none')(F.log_softmax(logits_context, dim=-1), F.softmax(logits, dim=-1)).sum() * 3.5 # self.args.scl_loss
            return logits, (kl_loss, scl_loss)

    @torch.no_grad()
    def set_classifier(self):
        text_prompt, tokenized_prompts = self.prompt_learner()
        try:
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
        except:
            text_features = []
            batch_size= 1000
            for bi in range(text_prompt.shape[0]//batch_size):
                batch_text_features = self.text_encoder(text_prompt[bi*1000:(bi+1)*1000], tokenized_prompts[bi*1000:(bi+1)*1000])
                text_features.append(batch_text_features)
            text_features = torch.cat(text_features, dim=0)
        n_dim = text_features.shape[-1]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_class, -1)

        # text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

    @property #变成属性
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype #return int/float


class CoOpAdapter:
    def __init__(self, args, n_ctx=16, use_float32=False, use_grad_checkpoint=False):
        self.args = args
        clip_model, _ = load(args.ckpt_path, device=f"cuda:{args.default_gpu}")
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint

        self.n_ctx = n_ctx # n_ctx 输入词数
        self.lr = args.lr*args.train_batch/20
        self.wd = args.wd # wd ??
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.args = args
        self.current_class_names = []

    @staticmethod
    def ortho_penalty(t):
        return ((t @t.T - torch.eye(t.shape[0]).to(t.device))**2).mean()

    def fit(self, data):
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

        self.model.eval()

        for epoch in tqdm(range(self.epochs)):
            for idx, (x, y, index) in tqdm(enumerate(train_loader), total=len(train_loader), desc = 'Training'):

                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)

                output, (kl_loss, scl_loss) = self.model(x.cuda(device=self.args.default_gpu))
                # pdb.set_trace()
                loss = F.cross_entropy(output, 
                                       y.cuda(device=self.args.default_gpu)) 
                # loss = F.cross_entropy(output.view(-1, output.shape[-1]), 
                #                        y.cuda(device=self.args.default_gpu).unsqueeze(0).expand(self.args.forward_times, -1).contiguous().view(-1)) 
                loss = loss + kl_loss + scl_loss
                # loss = loss + self.ortho_penalty(self.model.prompt_learner.ctx)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # pdb.set_trace()
            # print(self.model.prompt_learner.ctx)
            # print(self.model.image_encoder.layer1[0].conv1.weight[0])
        self.model.set_classifier()
        return self.model
 


    def init_model(self, class_names, per_epoch_steps, prompt_templates):

        self.n_class = len(class_names)
        clip_model = deepcopy(self.clip_model)

        self.model = CLIP(self.args, class_names, clip_model, self.n_ctx, prompt_templates)
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True 
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True

        param_dict = [{'params': [p for p in self.model.prompt_learner.parameters() if p.requires_grad] 
                       + [p for p in self.model.mu_adapter.parameters()]}]
        self.optimizer = torch.optim.SGD(param_dict, lr=self.lr, weight_decay=self.wd)
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=self.lr,
            total_step=self.epochs*per_epoch_steps)

    @torch.no_grad()
    def inference(self,image):
        logits = self.model(image, test=True)
        return logits.float().softmax(dim=-1)

    @torch.no_grad()
    def accuracy(self, loader, num_test=None, test_class=None, mean_per_class=False):
        if mean_per_class:
            return self._accuracy_mpc(loader)
        else:
            return self._accuracy(loader)

    def _accuracy_mpc(self, loader):
        n_class = self.n_class
        acc_per_class = [0 for _ in range(n_class)]
        count_per_class = [0 for _ in range(n_class)]
        for i, (x, y, _) in tqdm(enumerate(loader), total=len(loader), desc = 'running inference'):
            pred_y = self.inference(x.cuda(device=self.args.default_gpu))
            _, top_labels = pred_y.topk(1, dim=-1)
            for c in range(n_class):
                acc_per_class[c] += ((top_labels.view(-1) == y.cuda(device=self.args.default_gpu)) * (y.cuda(device=self.args.default_gpu)== c)).sum().item()
                count_per_class[c] += (y.cuda(device=self.args.default_gpu) == c).sum().item()
        acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()
        return acc

    def _accuracy(self, loader):
        total_count=0
        acc_count =0
        # pdb.set_trace()
        for i, (x, y, _) in tqdm(enumerate(loader), total=len(loader), desc = 'running inference'):
            pred_y = self.inference(x.cuda(device=self.args.default_gpu))
            _, top_labels = pred_y.topk(1, dim=-1)
            acc_count += (top_labels.view(-1)==y.cuda(device=self.args.default_gpu)).sum().cpu().numpy()
            total_count += y.shape[0]
        acc = acc_count*1.0/total_count
        acc = acc.item()
        return acc