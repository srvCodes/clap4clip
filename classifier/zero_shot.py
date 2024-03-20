import torch
import torch.nn as nn 
import numpy as np 
from tqdm import tqdm

from clip.clip import load, tokenize
from .evaluator import Evaluator

# import open_clip

    
class ZeroshotCLIP(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.clip_model, _ = load(args.ckpt_path, device=f"cuda:{args.default_gpu}")
        self.clip_model = self.clip_model.eval()
        self.current_class_names = []

    @torch.no_grad()
    def fit(self, data):
        self.current_class_names += data['class_names']
        print(f"Class names: {self.current_class_names}")
        self.n_class = len(self.current_class_names)
        prompts = [[temp.format(c.replace("_", " ")) for temp in data['prompt_templates'] ] for c in self.current_class_names]
        self.text_features = []
        with torch.no_grad():
            for per_cls_prompts in prompts:
                per_cls_prompt_embs = tokenize(per_cls_prompts).cuda(device=self.args.default_gpu)
                text_features = self.clip_model.encode_text(per_cls_prompt_embs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0)
                text_features = text_features / text_features.norm()
                self.text_features.append(text_features)
        self.text_features = torch.stack(self.text_features, dim=0)
        # if self.args.sess == 2:
        #     self.tsne_plot_text_features()
        # print(self.text_features.shape)
    
    def tsne_plot_text_features(self):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt 
        import seaborn as sns 
        plt.rcParams.update({'font.size': 18})

        taskwise_means = self.text_features.view(10, -1, 512).mean(0)
        to_plot = taskwise_means @ taskwise_means.t()
        ax = sns.displot(to_plot.detach().cpu().numpy(),  kind="kde", bw_adjust=.25,  aspect=1.7, linewidth=3, fill=True, common_norm=True, palette=['red', 'deepskyblue', 'orange'], legend=False)
        ax.set(xticklabels=[], yticklabels=[])
        ax.set(xlabel=None, ylabel=None)
        ax.tick_params(bottom=False, left=False)  # remove the ticks
        plt.legend(title='Task', labels=['1', '2', 't'])
        plt.tight_layout()
        # plt.axis('off')
        plt.savefig("distributions1.png")
        plt.show()
        pass 

    @torch.no_grad()
    def inference(self,image, label, num_test=None, test_class=None):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            logits = image_features @ self.text_features.t() * logit_scale
            if self.args.compute_ram:
                samplewise_text_feats = self.text_features[label]
                return logits.float().softmax(dim=-1), (image_features.detach().cpu(), samplewise_text_feats.detach().cpu())
        return logits.float(), (None, None)

    # def accuracy(self, loader, num_test=None, test_class=None, mean_per_class=False):
    #     total_count=0
    #     acc_count=0

    #     if mean_per_class:
    #         n_class = self.text_features.shape[0]
    #         acc_per_class = [0 for _ in range(n_class)]
    #         count_per_class = [0 for _ in range(n_class)]

    #     for i, (x, y, _) in tqdm(enumerate(loader), total=len(loader), desc = 'Running zero-shot inference..'):
    #         pred_y = self.inference(x.cuda())
    #         _, top_labels = pred_y.topk(1, dim=-1)

    #         if not mean_per_class:
    #             acc_count += (top_labels.view(-1)==y.cuda()).sum().cpu().numpy()
    #             total_count += y.shape[0]
    #         else:
    #             for c in range(n_class):
    #                 acc_per_class[c] += ((top_labels.view(-1) == y.cuda()) * (y.cuda()== c)).sum().item()
    #                 count_per_class[c]+=(y.cuda()==c).sum().item()

    #     if not mean_per_class:
    #         acc = acc_count*1.0/total_count
    #         acc = acc.item()
    #     else:
    #         acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
    #         acc = np.array(acc).mean()

    #     return acc