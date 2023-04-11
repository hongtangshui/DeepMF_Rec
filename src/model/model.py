import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


from ..bert_config.configuration_bert import BertConfig
from ..utils.attention import *
from ..utils.pooling import BertPooler
from ..utils.dropout import DropoutWrapper

from .ebd_prediction import NewsIdEmbedding, UserIdEmbedding
from .repr_similarity import IdSimilarity


'''
Now we use Matrix Factorization to train the user and news ID embedding,
but for cold start user, we only use his warm hist news to predict it user ID embedding
however, there may exist a following path:
    cold_start_user -- cold_start_news -- warm_user[has trained ID embedding]
In this case, we can user warm_user's trained ID embedding to predict the cold_start_user ID embedding.


We can span the current idea to GNN. We can construct a graph contains both trained nodes(users and news) and untrained nodes,
then use a special initilazation method(global avg pool) and GNN to spread the trained nodes info to untrained nodes.
But notice, this method could just predict ID embedding for [hist_news] and [user] in test set, since we don't know
the a candidates news in clicked by a user or not, which is the infomation we need to make prediction. 
'''


class DeepMatrixFactorization(nn.Module):
    '''
    Deep Matrix Factorization generate ID embedding of user and news in trainng set.
    Use two id embedding prediction module to handle cold start issue.
    '''
    def __init__(self, embedding_size, news_card, user_card, pretrained, device, args):
        super(DeepMatrixFactorization, self).__init__()
        '''
        Args:
            embedding_size: 768
            news_card: cardinality of news that has appeared in the training set 
            user_card: cardinality of user that has appeared in the training set
        '''
        self._pretrained = pretrained
        
        # one more embedding for cold start item / user
        self.news_embedding = nn.Embedding(news_card+1, embedding_size)
        self.user_embedding = nn.Embedding(user_card+1, embedding_size)

        self.news_predicted_embedding = NewsIdEmbedding()
        self.user_predicted_embedding = UserIdEmbedding()

        self._config = BertConfig.from_pretrained(self._pretrained)
        self.curr_encoder = BertModel.from_pretrained(self._pretrained)
        self.bertpooler = BertPooler(self._config)
        self.dropout = DropoutWrapper(dropout_p=0.1)

        self.id_similarity = IdSimilarity()

        self.user_warm_proportion = 1 / 10
        self.hist_warm_proportion = 2 / 3
        self.curr_warm_proportion = 1 / 2

    def pooler(self, input_sequence, attention_mask, news_mode):
        input_sequence = self.dropout(input_sequence)
        if news_mode == 'cls':
            token_embeddings = input_sequence
            return self.bertpooler(token_embeddings)
        elif news_mode == 'max':
            token_embeddings = input_sequence
            input_mask_expanded = attention_mask.unsqueeze(
                -1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = 1e-9
            max_over_time = torch.max(token_embeddings, 1)[0]
            return max_over_time
        elif news_mode == 'mean':
            token_embeddings = input_sequence
            input_mask_expanded = attention_mask.unsqueeze(
                -1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = 1e-9
            mean_over_time = torch.max(token_embeddings, 1)[0]
            return mean_over_time
        elif news_mode == 'attention':
            # TODO
            token_embeddings = input_sequence
            return token_embeddings
        else:
            raise NotImplementedError

    def forward(self, curr_input_ids, curr_token_type, curr_input_mask, curr_category_ids,
                hist_input_ids, hist_token_type, hist_input_mask, hist_mask, hist_category_ids,
                curr_idx, hist_idx, user_idx,
                curr_cold_mask, hist_cold_mask, user_cold_mask,
                ctr, recency):
        '''
        different processing for training and testing:
        train:
            use (hist_idx, user_idx) and (curr_idx, user_idx) to trainng the embedding
            train the hist_news -> user_idx to train MLP(hist_news, user_idx) 
            # TODO: the above two training process is contrastive, so we use mask ? or split one epoch to two phase
            train a MLP to estimate ID embedding for curr_idx   # MASK
        test:
            if user_idx is in user_embedding_idx(that is to as the current user is in training set):
                user ID embedding = user_embedding[user_idx]
            else:
                user ID embedding = MLP(hist_idx)
            if curr_idx is in item_embedding_idx:
                curr item ID embediing = item_embedding[curr_idx]
            else:
                curr ID embedding = MLP(curr_input_ids(BERT))


        Input:
            curr_input_ids, curr_type_ids, curr_input_mask: for BERT
            hist_input_ids, hist_token_type, hist_mask: for BERT
            curr_idx, hist_idx, user_idx: id
            user_cold_mask, curr_cold_mask, hist_cold_mask: if mask == 0, then the corresponding user/news's id embedding is not trained.
        '''
        if self.training:
            # By statistics, in test set, 1/10 user's id embedding is trained, so mask 9/10 of the user: user_cold_mask
            # 1/2 news's id embedding is trained, so mask 1/2 of the curr_user: curr_cold_mask
            # 2/3 hist news id embedding is trained, so mask 1/3 of the hist_news: hist_cold_mask
            user_cold_mask = torch.zeros_like(user_cold_mask)
            user_cold_mask[torch.rand(user_cold_mask.shape) < self.user_warm_proportion] = 1
            curr_cold_mask = torch.zeros_like(curr_cold_mask)
            curr_cold_mask[torch.rand(curr_cold_mask.shape) < self.curr_warm_proportion] = 1
            hist_cold_mask = torch.zeros_like(hist_cold_mask)
            hist_cold_mask[torch.rand(hist_cold_mask.shape) < self.hist_warm_proportion] = 1
            pass
        batch_size = curr_input_ids.shape[0]
        curr_num = curr_input_ids.shape[1]

        # calculate bert output of the current news for news ID embedding module to predict their ID embedding
        curr_input_ids = curr_input_ids.view(batch_size * curr_num, -1)
        curr_token_type = curr_token_type.view(batch_size * curr_num, -1)
        curr_input_mask = curr_input_mask.view(batch_size * curr_num, -1)
        curr_pooled_output = self.curr_encoder(
            input_ids=curr_input_ids,
            token_type_ids=curr_token_type,
            attention_mask=curr_input_mask,
        )[0]
        curr_pooled_output = self.pooler(curr_pooled_output, curr_input_mask, news_mode='cls')
        curr_pooled_output = self.dropout(curr_pooled_output).view(batch_size, curr_num, -1)

        # cal hist news's id embedding for user ID embedding module to predict ID embedding of users.
        hist_id_embedding = self.news_embedding(hist_idx)

        # cal curr news id embedding
        curr_id_embedding_trained = self.news_embedding(curr_idx)
        curr_id_embedding_predicted = self.news_predicted_embedding(curr_pooled_output)
        curr_id_embedding_trained = curr_id_embedding_trained.masked_fill_(~curr_cold_mask.unsqueeze(-1), 0.0)
        curr_id_embedding_predicted = curr_id_embedding_predicted.masked_fill_(curr_cold_mask.unsqueeze(-1), 0.0)
        curr_id_embedding = curr_id_embedding_predicted + curr_id_embedding_trained

        # cal user id embedding
        user_id_embedding_trained = self.user_embedding(user_idx)
        user_id_embedding_predicted = self.user_predicted_embedding(hist_id_embedding, hist_cold_mask)
        user_id_embedding_trained = user_id_embedding_trained.masked_fill_(~user_cold_mask.unsqueeze(-1), 0.0)
        user_id_embedding_predicted = user_id_embedding_predicted.masked_fill_(user_cold_mask.unsqueeze(-1), 0.0)
        user_id_embedding = user_id_embedding_predicted + user_id_embedding_trained

        # cal similarity
        id_score = self.id_similarity(user_id_embedding, curr_id_embedding)
        if self.training:
            hist_id_score = self.id_similarity(user_id_embedding, hist_id_embedding)
            return hist_id_score.squeeze(1), id_score.squeeze(1)

        return id_score
