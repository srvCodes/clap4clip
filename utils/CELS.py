import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, torch.unsqueeze(targets, 1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = torch.mean(torch.sum(-targets * log_probs, 1))
    return loss
         