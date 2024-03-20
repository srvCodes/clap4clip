import random
import time
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Lambda
from tqdm import tqdm
from classifier.utils import SubsetRandomSampler
from scipy.stats.mstats import gmean 

class ExemplarSelector():
    def __init__(self, args) -> None:
        self.args = args

    def set_dataset_and_transform(self, dataset, transform):
        self.train_dataset = dataset
        self.val_transform = transform

class RandomExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on random selection, which produces a random list of samples."""

    def __init__(self, args):
        super().__init__(args)

    def select_indices(self, model, exemplars_per_class, memory, for_memory) -> Tuple:

        data_memory_, targets_memory_ = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if(memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task*(self.args.sess)):
                idx = np.where(targets_memory==class_idx)[0][:exemplars_per_class]   #若task=2 取前100/class训练集图片
                data_memory_ = np.concatenate([data_memory_, np.tile(data_memory[idx], (mu,))   ])
                targets_memory_ = np.concatenate([targets_memory_, np.tile(targets_memory[idx], (mu,))    ])
                
                
        #add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task*(self.args.sess),self.args.class_per_task*(1+self.args.sess)):
            idx = np.where(new_targets==class_idx)[0][:exemplars_per_class]
            data_memory_ = np.concatenate([data_memory_, np.tile(new_indices[idx],(mu,))   ])
            targets_memory_ = np.concatenate([targets_memory_, np.tile(new_targets[idx],(mu,))    ])
        return data_memory_, targets_memory_

    def _get_labels(self, sel_loader):
        if hasattr(sel_loader.dataset, 'targets'):  # BaseDataset, MemoryDataset
            labels = np.asarray(sel_loader.dataset.targets)
        else:
            raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
        return labels


class HerdingExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def select_indices(self, model, exemplars_per_class: int, memory, for_memory) -> Tuple:
        all_memory_indices = np.concatenate([np.tile(memory[0], (1,)), for_memory[0]]) if memory is not None else for_memory[0]
        sel_loader =  torch.utils.data.DataLoader(self.train_dataset, 
                                                  batch_size=self.args.train_batch, 
                                                  shuffle=False, 
                                                  num_workers=4, 
                                                  sampler=SubsetRandomSampler(all_memory_indices, False))

        # extract outputs from the model for all train samples
        extracted_features = []
        extracted_targets = []
        extracted_indices = []
        with torch.no_grad():
            model.eval()
            for _, (images, targets, idx) in tqdm(enumerate(sel_loader), total=len(sel_loader), desc = 'Extracting exemplar features..'):
                feats, _ = model(images.cuda(device=self.args.default_gpu), test=True, )
                feats = feats / feats.norm(dim=1).view(-1, 1)  # Feature normalization
                extracted_features.append(feats)
                extracted_targets.extend(targets)
                extracted_indices.extend(idx)
       
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = np.array(extracted_targets)
        extracted_indices = np.array(extracted_indices)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # calculate the mean
            cls_mu = cls_feats.mean(0)
            # select the exemplars closer to the mean of each class
            selected = []
            selected_feat = []
            for k in range(exemplars_per_class):
                # fix this to the dimension of the model features
                sum_others = torch.zeros(cls_feats.shape[1])
                for j in selected_feat:
                    sum_others += j / (k + 1)
                dist_min = np.inf
                # choose the closest to the mean of the current class
                for item in cls_ind:
                    if item not in selected:
                        feat = extracted_features[item]
                        dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                        if dist < dist_min:
                            dist_min = dist
                            newone = item
                            newonefeat = feat
                selected_feat.append(newonefeat)
                selected.append(newone)
            result.extend(selected)
        data_memory_, targets_memory_ = extracted_indices[result], extracted_targets[result]

        return data_memory_, targets_memory_


class EntropyExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on entropy selection, which produces a sorted list of samples of one
    class based on entropy of each sample. From RWalk http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def select_indices(self, model, exemplars_per_class: int, memory, for_memory) -> Tuple:
        all_memory_indices = np.concatenate([np.tile(memory[0], (1,)), for_memory[0]]) if memory is not None else for_memory[0]
        sel_loader =  torch.utils.data.DataLoader(self.train_dataset, 
                                                  batch_size=self.args.train_batch, 
                                                  shuffle=False, 
                                                  num_workers=4, 
                                                  sampler=SubsetRandomSampler(all_memory_indices, False))

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        extracted_indices = []
        with torch.no_grad():
            model.eval()
            for _, (images, targets, idx) in tqdm(enumerate(sel_loader), total=len(sel_loader), desc = 'Extracting exemplar features..'):
                logits, _ = model(images.cuda(device=self.args.default_gpu), test=True)
                extracted_logits.append(logits.detach())
                extracted_targets.extend(targets)
                extracted_indices.extend(idx)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_targets = np.array(extracted_targets)
        extracted_indices = np.array(extracted_indices)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            # select the exemplars with higher entropy (lower: -entropy)
            probs = torch.softmax(cls_logits.float(), dim=1)
            log_probs = torch.log(probs)
            minus_entropy =  (probs * log_probs).sum(1)  # change sign of this variable for inverse order
            selected = cls_ind[minus_entropy.sort()[1][:exemplars_per_class]]
            result.extend(selected)

        data_memory_, targets_memory_ = extracted_indices[result], extracted_targets[result]
        return data_memory_, targets_memory_

class EnergyExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on entropy selection, which produces a sorted list of samples of one
    class based on entropy of each sample. From RWalk http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def select_indices(self, model, exemplars_per_class: int, memory, for_memory) -> Tuple:
        all_memory_indices = np.concatenate([np.tile(memory[0], (1,)), for_memory[0]]) if memory is not None else for_memory[0]
        sel_loader =  torch.utils.data.DataLoader(self.train_dataset, 
                                                  batch_size=self.args.train_batch, 
                                                  shuffle=False, 
                                                  num_workers=4, 
                                                  sampler=SubsetRandomSampler(all_memory_indices, False))

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        extracted_indices = []
        with torch.no_grad():
            model.eval()
            for _, (images, targets, idx) in tqdm(enumerate(sel_loader), total=len(sel_loader), desc = 'Extracting exemplar features..'):
                logits, _ = model(images.cuda(device=self.args.default_gpu), test=True, return_mean=False)
                extracted_logits.extend(logits.detach())
                extracted_targets.extend(targets)
                extracted_indices.extend(idx)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_logits = extracted_logits.reshape(len(extracted_targets), self.args.forward_times,  -1)
        extracted_targets = np.array(extracted_targets)
        extracted_indices = np.array(extracted_indices)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            energy_scores = torch.logsumexp(cls_logits.float(), dim=-1).mean(1)
            selected = cls_ind[energy_scores.sort()[1][:exemplars_per_class]]

            # probs = probs.mean(1)
            # log_probs = torch.log(probs)
            # minus_entropy = (probs * log_probs).sum(1)  # try with reverse symbol with both comibation of variance symbols

            # total = torch.stack([vars, minus_entropy], 0).numpy()
            # geo_mean = gmean(total)
            # selected = cls_ind[geo_mean.argsort()[:exemplars_per_class]]
            result.extend(selected)

        data_memory_, targets_memory_ = extracted_indices[result], extracted_targets[result]
        return data_memory_, targets_memory_


class VarianceExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on entropy selection, which produces a sorted list of samples of one
    class based on entropy of each sample. From RWalk http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def select_indices(self, model, exemplars_per_class: int, memory, for_memory) -> Tuple:
        all_memory_indices = np.concatenate([np.tile(memory[0], (1,)), for_memory[0]]) if memory is not None else for_memory[0]
        sel_loader =  torch.utils.data.DataLoader(self.train_dataset, 
                                                  batch_size=self.args.train_batch, 
                                                  shuffle=False, 
                                                  num_workers=4, 
                                                  sampler=SubsetRandomSampler(all_memory_indices, False))

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        extracted_indices = []
        with torch.no_grad():
            model.eval()
            for _, (images, targets, idx) in tqdm(enumerate(sel_loader), total=len(sel_loader), desc = 'Extracting exemplar features..'):
                logits, _ = model(images.cuda(device=self.args.default_gpu), test=True, return_mean=False)
                extracted_logits.extend(logits.detach())
                extracted_targets.extend(targets)
                extracted_indices.extend(idx)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_logits = extracted_logits.reshape(len(extracted_targets), self.args.forward_times,  -1)
        extracted_targets = np.array(extracted_targets)
        extracted_indices = np.array(extracted_indices)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            probs = torch.softmax(cls_logits.float(), dim=-1)
            vars = probs.var(1).sum(1)

            selected = cls_ind[vars.sort()[1][:exemplars_per_class]]

            # probs = probs.mean(1)
            # log_probs = torch.log(probs)
            # minus_entropy = (probs * log_probs).sum(1)  # try with reverse symbol with both comibation of variance symbols

            # total = torch.stack([vars, minus_entropy], 0).numpy()
            # geo_mean = gmean(total)
            # selected = cls_ind[geo_mean.argsort()[:exemplars_per_class]]
            result.extend(selected)

        data_memory_, targets_memory_ = extracted_indices[result], extracted_targets[result]
        return data_memory_, targets_memory_
    
class VarianceEntropyExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on entropy selection, which produces a sorted list of samples of one
    class based on entropy of each sample. From RWalk http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def select_indices(self, model, exemplars_per_class: int, memory, for_memory) -> Tuple:
        all_memory_indices = np.concatenate([np.tile(memory[0], (1,)), for_memory[0]]) if memory is not None else for_memory[0]
        sel_loader =  torch.utils.data.DataLoader(self.train_dataset, 
                                                  batch_size=self.args.train_batch, 
                                                  shuffle=False, 
                                                  num_workers=4, 
                                                  sampler=SubsetRandomSampler(all_memory_indices, False))

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        extracted_indices = []
        with torch.no_grad():
            model.eval()
            for _, (images, targets, idx) in tqdm(enumerate(sel_loader), total=len(sel_loader), desc = 'Extracting exemplar features..'):
                logits, _ = model(images.cuda(device=self.args.default_gpu), test=True, return_mean=False)
                extracted_logits.extend(logits.detach())
                extracted_targets.extend(targets)
                extracted_indices.extend(idx)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_logits = extracted_logits.reshape(len(extracted_targets), self.args.forward_times,  -1)
        extracted_targets = np.array(extracted_targets)
        extracted_indices = np.array(extracted_indices)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            probs = torch.softmax(cls_logits.float(), dim=-1)
            vars = probs.var(1).sum(1)

            probs = probs.mean(1)
            log_probs = torch.log(probs)
            minus_entropy = (probs * log_probs).sum(1)  # try with reverse symbol with both comibation of variance symbols

            total = torch.stack([vars, minus_entropy], 0).numpy()
            geo_mean = gmean(total)
            selected = cls_ind[geo_mean.argsort()[:exemplars_per_class]]
            result.extend(selected)

        data_memory_, targets_memory_ = extracted_indices[result], extracted_targets[result]
        return data_memory_, targets_memory_

class DistanceExemplarsSelector(ExemplarSelector):
    """Selection of new samples. This is based on distance-based selection, which produces a sorted list of samples of
    one class based on closeness to decision boundary of each sample. From RWalk
    http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, args):
        super().__init__(args)


    @torch.no_grad()
    def select_indices(self, model, exemplars_per_class: int, memory, for_memory) -> Tuple:
        all_memory_indices = np.concatenate([np.tile(memory[0], (1,)), for_memory[0]]) if memory is not None else for_memory[0]
        sel_loader =  torch.utils.data.DataLoader(self.train_dataset, 
                                                  batch_size=self.args.train_batch, 
                                                  shuffle=False, 
                                                  num_workers=4, 
                                                  sampler=SubsetRandomSampler(all_memory_indices, False))

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        extracted_indices = []
        with torch.no_grad():
            model.eval()
            for _, (images, targets, idx) in tqdm(enumerate(sel_loader), total=len(sel_loader), desc = 'Extracting exemplar features..'):
                logits, _ = model(images.cuda(device=self.args.default_gpu), test=True)
                extracted_logits.append(logits.detach())
                extracted_targets.extend(targets)
                extracted_indices.extend(idx)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_targets = np.array(extracted_targets)
        extracted_indices = np.array(extracted_indices)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            # select the exemplars closer to boundary
            distance = cls_logits[:, curr_cls]  # change sign of this variable for inverse order
            selected = cls_ind[distance.sort()[1][:exemplars_per_class]]
            result.extend(selected)
        data_memory_, targets_memory_ = extracted_indices[result], extracted_targets[result]
        return data_memory_, targets_memory_