import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.attention import *


class UserIdEmbedding(nn.Module):
    '''
    use a user's hist news ID embedding to predict user's embedding
    '''
    def __init__(self, ):
        super(UserIdEmbedding, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim=768, head_num=24)
        self.projection = nn.Linear(768, 768)
    
    def forward(self, hist_embedding, hist_mask):
        '''
        Use avg pooling to get user ID embedding by users history news ID embedding
        
        Why not attention ? : attention is too complex, making the loss don't converage.

        Input:
            hist_embedding(torch.Tensor): [batch_size, hist_len, hidden_dim]. hist news ID embedding.
            hist_mask(torch.Tensor): [batch_size, hist_len]. if hist_mask == 0, than the corresponding hist news is a cold-start hist news,
                otherwise it's a warm hist news which has trained news ID embedding.
        Returns:
            user_embedding(torch.Tensor): [batch_size, hidden_size]. 

        '''
        # self attention + avg pool
        # attention_mask = torch.matmul(hist_mask.unsqueeze(-1).to(torch.float), 
        #                          hist_mask.unsqueeze(-1).to(torch.float).transpose(-2, -1)) > 0.5    # [batch_size, hist_len, hist_len] 
        # context, weight = self.attention(hist_embedding, hist_embedding, hist_embedding, attention_mask) # [batch_size, hist_len, hidden_dim]
        # context = context.masked_fill_(~hist_mask.unsqueeze(-1), 0.0)
        # context = torch.mean(context, dim=-2)   # [batch_size, hidden_dim]

        # avg_pool
        context = hist_embedding.masked_fill_(~hist_mask.unsqueeze(-1), 0.0)
        context = torch.mean(context, dim=1)
        context = F.gelu(self.projection(context))
        return context


class NewsIdEmbedding(nn.Module):
    '''
    use news content to predict news ID embedding for cold start candidates news
    '''
    def __init__(self, ):
        super(NewsIdEmbedding, self).__init__()
        hidden_size = 768
        self.project1 = nn.Linear(hidden_size, hidden_size)
        self.project2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, curr_pooled_output):
        '''
        Input:
            curr_pooled_output (torch.Tensor): [batch_size, curr_len(5), hidden_size(768)]: bert's output
        Output:
            curr_id_embedding (torch.Tensor): [batch_size, curr_len(5), hidden_size(768)]: predicted news ID embedding
        '''
        curr_id_embedding = F.gelu(self.project1(curr_pooled_output))
        curr_id_embedding = F.gelu(self.project2(curr_id_embedding))
        return curr_id_embedding
