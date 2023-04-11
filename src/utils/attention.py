import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data



class DotProductAttention(nn.Module):
    '''
    Dot product: user query * key as weight
    '''
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # query [batch_size, q_len, hidden_dim]
        # kay [batch_size, v_len, hidden_dim]
        v_len = query.shape[1]
        batch_size, v_len, hidden_dim = key.shape

        score = torch.bmm(query, key.permute(0, 2, 1))
        weight = F.softmax(score, dim=-1)
        weight = self.dropout(weight)

        context = torch.bmm(weight, value)

        return context, weight

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot product attention.
    weight = (q*v)/sqrt(dim) 
    '''
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        '''
        Args:
            dim (int): hidden dimension
        '''
        self.sqrt_dim = np.sqrt(dim)
    
    def forward(self, query, key, value, mask):
        '''
        Args:
            q:      [batch_size, q_len, hidden_dim]
            k:      [batch_size, v_len, hidden_dim]
            v:      [batch_size, v_len, hidden_dim]
            mask:   [batch_size, q_len, v_len], exclude some interaction between key(value) and key(value) from attention
    
        Returns:
            context (torch.Tensor): [batch_size, q_len, hidden_dim], result of the attention mechanism
            weight (torch.Tensor): [batch_size, q_len, hidden_dim], attention weight
        '''
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(~mask, -float("Inf"))
        weight = F.softmax(score, -1)
        context = torch.bmm(weight, value)
        return context, weight

class MultiHeadAttention(nn.Module):
    '''
    MutiHead Attention
    '''
    def __init__(self, hidden_dim, head_num):
        super(MultiHeadAttention, self).__init__()
        '''
        Args:
            hidden_dim(int): hidden_dim
            head_num(int): num of heads
        '''
        self.head_num = head_num
        assert hidden_dim % head_num == 0, "hidden_dim {} must divide head_num {}".format(hidden_dim, head_num)
        self.head_dim = hidden_dim // head_num
        self.hidden_dim = hidden_dim

        self.k_project = nn.Linear(hidden_dim, self.head_num * self.head_dim)
        self.v_project = nn.Linear(hidden_dim, self.head_num * self.head_dim)
        self.q_project = nn.Linear(hidden_dim, self.head_num * self.head_dim)

        nn.init.xavier_normal_(self.k_project.weight)
        nn.init.xavier_normal_(self.v_project.weight)
        nn.init.xavier_normal_(self.q_project.weight)
    
    def transpose_for_scores(self, x):
        '''
        Split the input into multi head
        Args:
            x(torch.tensor): [batch_size, len, hidden_dim]
        Return:
            transposed(torch.tensor): [batch_size, head_num, len, head_dim]
        '''
        new_x_shape = x.size()[:-1] + (self.head_num, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        '''
        Args:
            query(torch.Tensor): [batch_size, q_len, hidden_dim]
            key(torch.Tensor): [batch_size, v_len, hidden_dim]
            value(torch.Tensor): [batch_size, v_len, hidden_dim]
            mask(torch.Tensor): [batch_size, q_len, v_len], exculde some intersection between key(value) and key(value) from attention
        Returns:
            context(torch.Tensor): [batch_size, q_len, hidden_dim]
            weight(torch.Tensor): [batch_size, q_len, hidden_dim]
        '''
        context_shape = query.shape
        query   = self.transpose_for_scores(self.q_project(query))
        value   = self.transpose_for_scores(self.v_project(value))
        key     = self.transpose_for_scores(self.k_project(key))

        # [B, H, L1, D] + [B, H, L2, D] = [B, H, L1, L2]
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill_(~mask.unsqueeze(dim=1), -float("Inf"))
        weight = F.softmax(score, dim=-1) 
        context = torch.matmul(weight, value)   # [B, H, L2, D]
        context = context.permute(0, 2, 1, 3).reshape(context_shape)
        return context, weight


    

