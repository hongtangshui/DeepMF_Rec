import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f


class Loss(nn.Module):
    def __init__(self, criterion):
        super(Loss, self).__init__()
        self._criterion = criterion
        self._mseloss = nn.MSELoss(reduction='mean')

    @staticmethod
    def compute_eval_loss(poly_attn: Tensor, logit:Tensor, labels:Tensor):
        '''
        compute loss for evaluation phase
        
        Args:
            poly_attn (torch.Tensor): [batch_size, num_context_codes, embedding_dim]
            logits (torch.Tensor): [batch_size, 1]
            labels (torch.Tensor): [batch_size, 1]
        Return:
            loss: loss value
        '''
        pass

    def forward(self, logits: Tensor, hist_logits: Tensor, labels: Tensor, hist_mask: Tensor):
        '''
        compute loss for training phase
        Args:
            logits (torch.Tensor): [batch_size, npratio+1 ]
            hist_logits (torch.Tensor): [batch_size, hist_max_len]: []
            labels (torch.Tensor): [batch_size, npratio+1 ]
            hist_mask (torch.Tensor): [batch_size, hist_max_len]: [True, True, True, 0] 
        Return:
            total_loss (torch.Tensor): 
        '''
        targets = labels.argmax(dim=1)
        cand_loss = self._criterion(logits, targets)
        
        #hist_logits = hist_logits.masked_fill_(~hist_mask, 0.0)
        hist_mask = hist_mask.float()   # torch.float(hist_mask)
        hist_loss = self._mseloss(hist_logits, hist_mask)
        total_loss = cand_loss + hist_loss
        return cand_loss