import torch
import torch.nn as nn
import torch.nn.functional as F

class IdSimilarity(nn.Module):
    '''
    cal similarity between user ID embedding and news ID embedding
    '''
    def __init__(self, ):
        super(IdSimilarity, self).__init__()
        self.user_projection = nn.Linear(768, 768)
        self.news_projection = nn.Linear(768, 768)
        self.dropout_p = 0.1
    
    def forward(self, user_id_embedding, news_id_embedding):
        '''
        Args:
            user_id_embedding (torch.Tensor): [batch_size, hidden_dim]
            news_id_embedding (torch.Tensor): [batch_size, num, hidden_dim]
        
        Output:
            id_score (torch.Tensor): [batch_size, 1, num]
        '''
        user_id_embedding = user_id_embedding.unsqueeze(dim=1)
        
        user_repr = F.gelu(self.user_projection(user_id_embedding)) #[B, 1, H]
        news_repr = F.gelu(self.news_projection(news_id_embedding)) #[B, num, H]

        score = torch.matmul(user_repr, news_repr.transpose(-2, -1))
        return score
        
        