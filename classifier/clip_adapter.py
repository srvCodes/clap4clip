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


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model, temp=None):
        super().__init__()
        self.n_class = len(class_names)
        self.args = args
        self.clip_model = clip_model
        self.current_class_names = class_names

        # image encoder
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        ctx_dim = self.clip_model.ln_final.weight.shape[0]

        # clip adapter
        self.adapter = Adapter(ctx_dim, 4).cuda(device=self.args.default_gpu).type(self.clip_model.dtype)
        # prompt templaates
        self.temp = temp if temp is not None else ["A photo of a"]
        self.text_features = self.get_text_features()

    def get_text_features(self):
        prompts = [[template.format(c.replace("_", " ")) for template in self.temp] for c in self.current_class_names]
        self.text_features = []
        with torch.no_grad():
            for per_cls_prompts in prompts:
                per_cls_prompt_embs = tokenize(per_cls_prompts).cuda(device=self.args.default_gpu)
                text_features = self.clip_model.encode_text(per_cls_prompt_embs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0)
                text_features = text_features / text_features.norm()
                self.text_features.append(text_features)
        text_features = torch.stack(self.text_features, dim=0)
        return text_features

    
    def forward(self, image, label=None, test=False):

        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)
        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        n_class = self.n_class

        if test:
            text_features = self.text_features
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            if self.args.compute_ram:
                visual_feats = image_features
                textual_feats = text_features[label]
                return logits, (visual_feats.detach().cpu(), textual_feats.detach().cpu())
            return logits, (None, None)

        else:
            # text_features_all = []
            text_features = self.text_features
                # text_features_all.append(text_features)
            # text_features = torch.stack(text_features_all).sum(0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(n_class, -1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits

    @torch.no_grad()
    def set_classifier(self):
        pass 

    @property #变成属性
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype #return int/float


class ClipAdapter(Evaluator):
    def __init__(self, args, use_float32=False, use_grad_checkpoint=False):
        super().__init__(args)
        self.args = args
        clip_model, _ = load(args.ckpt_path, device=f"cuda:{args.default_gpu}")
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        
        self.lr = args.lr
        self.wd = args.wd 
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.current_class_names = []

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

        self.init_model(class_names=self.current_class_names, per_epoch_steps=per_epoch_steps, temp=data['prompt_templates'])

        self.model.eval()
        if self.args.sess >= 0:
            for epoch in tqdm(range(self.epochs)):
                for idx, (x, y, index) in tqdm(enumerate(train_loader), total=len(train_loader), desc = 'Training'):

                    cur_iter_idx = epoch*per_epoch_steps+idx
                    self.cur_iter_idx = cur_iter_idx
                    self.scheduler.step(cur_iter_idx)

                    output = self.model(x.cuda(device=self.args.default_gpu))
                    # pdb.set_trace()
                    loss = F.cross_entropy(output, y.cuda(device=self.args.default_gpu))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
        self.model.set_classifier()
        return self.model
 
    def finetuning(self, data):
        memory_loader = data['memory_loader']
        self.cur_iter_idx = 0

        if len(memory_loader.dataset)< self.train_batch:
            real_img_bsz = len(memory_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch

        per_epoch_steps = len(memory_loader)

        self.build_optimizer(per_epoch_steps=per_epoch_steps, lr=self.lr/10., finetune=True)

        self.model.eval()

        for epoch in tqdm(range(self.args.finetune_epochs)):
            for idx, (x, y, index) in tqdm(enumerate(memory_loader), total=len(memory_loader), desc = 'Finetuning'):

                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)

                output = self.model(x.cuda(device=self.args.default_gpu))
                # pdb.set_trace()
                loss = F.cross_entropy(output, y.cuda(device=self.args.default_gpu))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
       
        return self.model
    
   
    def post_training(self, finalize=False):
        self.model.set_classifier()

    def init_model(self, class_names, per_epoch_steps, temp=None):
        self.n_class = len(class_names)
        clip_model = deepcopy(self.clip_model)

        self.model = CLIP(self.args, class_names, clip_model, temp=temp)

        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True 
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True
        self.build_optimizer(per_epoch_steps, lr=self.lr, warmup=True)

    def build_optimizer(self, per_epoch_steps, lr, warmup=False, finetune=False):
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
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


