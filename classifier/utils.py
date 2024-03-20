import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans 
import pdb
import math 
import random
from torch.utils.data import Sampler
import torch.nn.functional as F 

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
def ce_loss_np(logits, targets_onehot, sample_T):
    
    pred = F.softmax(logits, dim=-1)
   
    B = pred.size(1)
    targets_onehot_expand = targets_onehot.unsqueeze(0).expand(sample_T, -1, -1) 
    loss =torch.sum(-targets_onehot_expand * pred.log())

    return loss/(B*sample_T)

def get_context_by_labels(labels, m=None):
    _, idx, counts = torch.unique(labels, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, idx_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]).to(cum_sum.device), cum_sum[:-1]))
    context_indices = idx_sorted[cum_sum]
    if m is not None and context_indices.size(0) < m:
        diff = m - context_indices.size(0)
        context_indices_permuted = torch.randperm(labels.size(0)).to(labels.device)
        context_indices_permuted = context_indices_permuted[(context_indices_permuted != context_indices.view(-1, 1)).all(dim=0)]
        context_indices = torch.cat((context_indices, context_indices_permuted[:diff]))
    return context_indices

def compute_uncertainty(logits, T=1.):
    logits = logits * T
    pseudo_label = torch.softmax(logits, dim=-1)
    if logits.dim() == 3:
        pseudo_label = pseudo_label.mean(0)
    uncertainty = torch.special.entr(pseudo_label).sum(1)
    return uncertainty

@torch.no_grad()
def get_context_indices_by_uncertainty(bs, labels, logits, task_specific_labels=None, top_k=1):
    unique_labels = torch.unique(labels)
    labels_to_indices = {label.item(): (labels == label).nonzero().flatten() for label in unique_labels}
    uncertainties = compute_uncertainty(logits) 
    uncertainties_by_labels = {label.item(): uncertainties[labels_to_indices[label.item()]] for label in unique_labels} 
    uncertainties_by_labels_sorted_indices = {label: labels_to_indices[label][torch.argsort(uncs, descending=False)]  for label, uncs in uncertainties_by_labels.items()}
    context_indices = torch.cat([indices[:top_k] for _, indices in uncertainties_by_labels_sorted_indices.items()])
    return context_indices.detach()

@torch.no_grad()
def get_context_indices( bs, labels, task_specific_labels=None, context_size=0.67):
    if task_specific_labels is None:
        # m = random.randint(math.ceil(0.3 * bs), math.ceil(0.8 * bs))
        m = math.ceil(context_size * bs)
        context_indices = torch.randperm(labels.size(0)).to(labels.device)[:m]
        # context_indices = get_context_by_labels(labels, m)
    else:
        context_indices = []
        for label in task_specific_labels:
            idx = (labels == label).nonzero(as_tuple=True)[0]
            context_indices.append(idx)
        context_indices = torch.cat(context_indices)
        if context_indices.shape[0] == labels.shape[0]:
            context_indices = get_context_indices(bs, labels)
    return context_indices

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, torch.nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad

def cosine_schedule_warmup(total_step, value, final_value=0, warmup_step=0, warmup_value=0):
    if warmup_step > 0:
        warmup_schedule = np.linspace(warmup_value, value, warmup_step+2)[1:-1]
    else:
        warmup_schedule = np.array([])
    steps = np.arange(total_step - warmup_step)
    schedule = final_value + 0.5 * (value-final_value) * (1+np.cos(np.pi * steps / len(steps)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_step
    return schedule

class build_cosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        init_lr = 0
        final_lr = lr * 1e-3
        self.lrs = cosine_schedule_warmup(total_step, lr, final_lr, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr = self.lrs[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"]= lr
        self.lr=lr

class build_bicosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        lr_promt = lr[0]
        lr_conv = lr[1]
        init_lr=0
        final_lr_promt = lr_promt * 1e-3
        final_lr_conv = lr_conv * 1e-3
        self.lrs_prompt = cosine_schedule_warmup(total_step, lr_promt, final_lr_promt, lr_warmup_step, init_lr)
        self.lrs_conv = cosine_schedule_warmup(total_step, lr_conv, final_lr_conv, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr_promt = self.lrs_prompt[idx]
        lr_conv = self.lrs_conv[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            # pdb.set_trace()
            if i==0:
                param_group["lr"] = lr_conv
            else:
                param_group["lr"] = lr_promt 
        self.lr_conv = lr_conv
        self.lr_prompt = lr_promt

def plot_tsne(features, labels, id):
    """
    features:(N*m)N*m大小特征,其中N代表有N个数据,每个数据m维
    label:(N)有N个标签
    """
    fig_path = "/home/ma-user/work/proda/visualization/tsne_{}.png".format(id)
    features = features.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # import pandas as pd
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # import seaborn as sns
    # class_num = len(np.unique(labels)）#要分类的种类个数 eg:[0,1,2,3]这个就是为4



    # tsne_features = tsne.fit_transform(features)#将特征使用PCA降维至2维
    # print('tsne_features的shape:',tsne_features.shape)
    # # plt.scatter(tsne_features[:, 0], tsne_features[:, 1])#将对降维的特征进行可视化
    # # plt.show()
    # plt.savefig(fig_path)

    # sns.set()
    # df = pd.DataFrame()
    # df["y"] = labels
    # df["comp-1"] = tsne_features[:,0]
    # df["comp-2"] = tsne_features[:,1]


    # fig = sns.scatterplot(x="comp-1", y="comp-2",hue=df.y.tolist(),
    #               palette=sns.color_palette("hls", class_num),
    #               data=df).set(title="Bearing data T-SNE projection")

    # scatter_fig = fig.get_figure()
    # scatter_fig.savefig(fig_path, dpi = 400)

    tSNE = TSNE()
    word_embeddings = tSNE.fit_transform(features)
    classifier = KMeans(n_clusters=len(np.unique(labels)))
    classifier.fit(word_embeddings)
    labels = classifier.labels_
    min_left = min(word_embeddings[:, 0])
    max_right = max(word_embeddings[:, 0])
    min_bottom = min(word_embeddings[:, 1])
    max_top = max(word_embeddings[:, 1])
    # markers = ["bo","go",,"mo","yo","ko","bx","gx", "rx"]
    colors =["b","g","r","y", "k", "slategrey","slateblue","pink"]
    marks = ["o","o","o","o","o","o","o","o","o","o","x","x","x","x","x","x","x","x","x","x"]
    for i in range(len(word_embeddings)):
        plt.plot(word_embeddings[i][0], word_embeddings[i][1], marker=marks[labels[i]], color=colors[labels[i]])
    plt.axis([min_left, max_right, min_bottom, max_top])
    plt.savefig(fig_path)
    plt.clf()


def plot_histogram(image1,image2,n):
    # image1 = image1.reshape(image1.shape[0],-1).cpu()
    # image2 = image2.reshape(image2.shape[0],-1).cpu()
    image1 = image1.reshape(-1).cpu()
    image2 = image2.reshape(-1).cpu()
    image3 = torch.cat((image1,image2),0).detach().numpy()
    image1 = image1.detach().numpy()
    imagez = image2.detach().numpy()
    # bins = np.linspace(image3.min(),image3.max(),n)
    bins = np.linspace(-0.045,0.045,n)
    # for i in range(image1.shape[0]):
    # pdb.set_trace()
    i = 0
    j = 8
    # plt.ylim((0,15000))
    plt.ylim((0,400))
    # plt.hist(image1[i], bins, alpha=0.5, label='x_1')
    # plt.hist(image1[j], bins, alpha=0.5, label='x_2')
    plt.hist(image1, bins, alpha=0.5, label='Image features')
    plt.hist(image2, bins, alpha=0.5, label='Text features')
    plt.legend(loc='upper right',fontsize=15)
    # print("image",image1[i].mean(),image1[j].mean(),image1[i].mean()-image1[j].mean())
    fig_path = "/home/ma-user/work/proda/visualization/histogram_kl.png"
    plt.savefig(fig_path)
    plt.clf()
    # plt.ylim((0,15000))
    # plt.hist(image2[i], bins, alpha=0.5, label='adv_1')
    # plt.hist(image2[j], bins, alpha=0.5, label='adv_2')
    # plt.legend(loc='upper right')
    # print("text",image2[i].mean(),image2[j].mean(),image2[i].mean()-image2[j].mean())
    # fig_path = "/home/ma-user/work/proda/visualization/histogram_text0.png"
    # plt.savefig(fig_path)
    # plt.clf()
    # pdb.set_trace()

def cosine_loss(q,k):
    # pdb.set_trace()
    q = q.repeat(1,k.shape[1],1)
    # k = k.squeeze(1)
    # q = q/q.norm(dim=-1)
    k_norm = k.norm(dim=-1,keepdim=True)
    # pdb.set_trace()
    # k_norm = k.norm(dim=-1).unsqueeze(1).repeat(1,k.shape[1])
    k = k/k_norm
    cos = ((q*k)/(k.shape[0]*k.shape[1])).sum()
    return 1-cos
