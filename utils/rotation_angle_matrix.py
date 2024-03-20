import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
import numpy as np



class RotationAngleMatrix():
    def __init__(self, args) -> None:
        self.args = args 
        self.task_to_sample_to_visual_feats = {}
        self.task_to_sample_to_textual_feats = {}
        self.task_to_sample_to_indices = {} # for verification, check if indices are same at each task
        self.label_range_to_select = np.arange(10)
        self.arccos_by_tasks_visual = {}
        self.arccos_by_tasks_textual = {}

    def get_relevant_logits(self, labels, logits_visual, logits_textual, sample_indices):
        relevant_logits_visual = []
        relevant_logits_textual = []
        relevant_indices = [] # for verification
        labels = np.array(labels)
        for curr_cls in self.label_range_to_select:
            cls_ind = np.where(labels == curr_cls)[0]
            relevant_logits_visual.append(logits_visual[cls_ind])
            relevant_logits_textual.append(logits_textual[cls_ind])
            relevant_indices.extend(sample_indices[cls_ind])

        relevant_indices = np.array(relevant_indices)
        sorting_indices = np.argsort(relevant_indices)
        relevant_logits_visual = np.array(torch.cat(relevant_logits_visual))[sorting_indices]
        relevant_logits_textual = np.array(torch.cat(relevant_logits_textual))[sorting_indices]
        relevant_indices = relevant_indices[sorting_indices]
        return relevant_logits_visual, relevant_logits_textual, relevant_indices
    

    def store_relevant_logits(self,  cur_task, labels, logits_visual, logits_textual, sample_indices):
        print(f"Storing logits ..")
        relevant_logits_visual, relevant_logits_textual, relevant_indices = self.get_relevant_logits(labels, logits_visual, logits_textual, sample_indices)
        self.task_to_sample_to_visual_feats[cur_task] = dict(zip(relevant_indices, relevant_logits_visual))
        self.task_to_sample_to_textual_feats[cur_task] = dict(zip(relevant_indices, relevant_logits_textual))
        
    def compute_arccos(self, task_a, task_b, mode='visual'):
        assert list(self.task_to_sample_to_visual_feats[task_a].keys()) == list(self.task_to_sample_to_visual_feats[task_b].keys()), \
        f"Test indices mismatch: {list(self.task_to_sample_to_visual_feats[task_a].keys())[:20]} vs {list(self.task_to_sample_to_visual_feats[task_b].keys())[:20]}!"
        
        if mode == 'visual':
            dot_prod = np.array(list(self.task_to_sample_to_visual_feats[task_a].values())) @ np.array(list(self.task_to_sample_to_visual_feats[task_b].values())).T
            dot_prod = np.clip(dot_prod, -1, 1)
            arccos = np.rad2deg(np.arccos(dot_prod)).mean()
            self.arccos_by_tasks_visual[(task_a, task_b)] = arccos
        elif mode == 'textual':
            dot_prod = np.array(list(self.task_to_sample_to_textual_feats[task_a].values())) @ np.array(list(self.task_to_sample_to_textual_feats[task_b].values())).T
            dot_prod = np.clip(dot_prod, -1, 1)
            arccos = np.rad2deg(np.arccos(dot_prod)).mean()
            self.arccos_by_tasks_textual[(task_a, task_b)] = arccos 
        else:
            raise NotImplementedError
        
    def compute_rotation_angle_matrix(self, cur_task, labels, logits_visual, logits_textual, sample_indices):
        self.store_relevant_logits(cur_task, labels, logits_visual, logits_textual, sample_indices)
        if cur_task > 0:
            self.compute_arccos(cur_task, 0, mode='visual')
            self.compute_arccos(cur_task, 0, mode='textual')
            print(f"RAM across visual: {self.arccos_by_tasks_visual}, textual: {self.arccos_by_tasks_textual}")
