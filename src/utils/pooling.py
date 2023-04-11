import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def scaled_attention(query, key, value, attn_mask=None):
    """
    Cal scaled attention output

    Args:
        query (torch.Tensor): [batch_size, *, q_len, key_dim]
        key (torch.Tensor): [batch_size, *, k_len, key_dim]
        value (torch.Tensor): [batch_size, *, k_len, value_dim]
        atte_mask (torch.Tensor): [batch_size, *, q_len, k_len]

    Returns:
        context (torch.Tensor): [batch_size, *, q_len, k_len]
    """
    assert query.shape[-1] == key.shape[-1]
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    weight = torch.softmax(score, dim=-1)
    context = torch.matmul(weight, value)
    return context

def get_attn_mask(attn_mask):
    '''
    extend the attn mask
    Args:
        attn_mask (torch.Tensor): [batch_size, *]
    
    Return:
        attn_mask (torch.Tensor): [batch_size, 1, *, *]
    '''
    if attn_mask is None:
        return None

    assert attn_mask.dim() == 2

    extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
    extended_attn_mask2 = extended_attn_mask.squeeze(-2).unsqueeze(-1)

    attn_mask = extended_attn_mask * extended_attn_mask2

    return attn_mask


class AttentionPooler(nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.query_news = nn.Parameter(torch.randn(1, manager.hidden_size))
        nn.init.xavier_normal_(self.query_news)
    
    def forward(self, news_reprs, hist_mask=None, *args, **kargs):
        '''
        encode user history into a representation vector

        Args:
            news_reprs(torch.Tensor): [batch_size, *, hidden_him] news representations
        Returns:
            user_reprs(torch.Tensor): [batch_size, 1, hidden_dim]  
        '''
        if hist_mask is not None:
            hist_mask = hist_mask.to(news_reprs.device).transpose(-1, -2)
        user_repr = scaled_attention(self.query_news, news_reprs, news_reprs, attn_mask=hist_mask)
        return user_repr

class BertPooler(nn.Module):
    '''
    the bert output corresponding to the [cls] token.
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        '''
        Args:
            hidden_states (torch.Tensor): [batch_size, length, hidden_dim]
        Output:
            pooled_output (torch.Tensor): [batch_size, 1, hidden_dim]
        '''
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
