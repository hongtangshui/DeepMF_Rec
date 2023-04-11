from typing import Dict, Any

import os
import random
import pickle
import numpy as np
import pandas as pd
import json
import ast

import torch
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

CATEGORY_CNT = 18

stop_words = set(stopwords.words('english'))
word_tokenizer = RegexpTokenizer(r'\w+')
pd.set_option('display.max_columns', None)


def remove_stopword(sentence):
    return ' '.join([word for word in word_tokenizer.tokenize(sentence) if word not in stop_words])

def union_list(l: list)->list:
    res = []
    for _l in l:
        res += l
    return res

def sampling(imps, ratio=4):
    pos = []
    neg = []
    res = ""
    for imp in imps.split():
        if imp[-1] == '1':
            pos.append(imp)
        else:
            neg.append(imp)
    n_neg = ratio
    if n_neg > len(neg):
        for idx in range(n_neg - len(neg)):
            neg.append(random.choice(neg))
    
    for p in pos:
        n = random.sample(neg, n_neg)
        t = [p] + n
        random.shuffle(t)
        if res == "":
            res = " ".join(t)
        else:
            res = res + "\t" + " ".join(t)
    return res

def get_recency_dict(x, y):
    return dict(zip(x.split(" "), y))

def get_recency(x, y):
    return [x[imp] for imp in y.split()]

def process_click(imps):
    click = []
    for imp in imps.split():
        imp_list = imp.split('-')
        click.append(int(imp_list[1]))
    return click

def process_news_id(imps):
    news_id = []
    for imp in imps.split():
        imp_list = imp.split('-')
        news_id.append(imp_list[0])
    return news_id        
    

class MindDataset(Dataset):
    def __init__(self, root: str, tokenizer: AutoTokenizer, mode: str,
                 split: str, hist_max_len: int, seq_max_len: int, data_type: str) -> None:
        super(MindDataset, self).__init__()
        self.data_path = os.path.join(root, split)
        self._mode = mode
        self._split = split
        self._data_type = data_type
        self._tokenizer = tokenizer
        self._hist_max_len = hist_max_len
        self._seq_max_len = seq_max_len

        self.category2id = self.read_json("category2id.json")
        self.subcategory2idx = self.read_json("subcategory2id.json")
        self.warm_news, self.warm_user = self.get_warm_data()

        self.news2idx = self.read_json("news2idx.json")
        self.user2idx = self.read_json("user2idx.json")
        self._news = self.process_news()
        self._examples = self.get_examples(negative_sampling=4)

    def get_warm_data(self):
        warm_news = set()
        warm_user = set()
        train_behavior_file = os.path.join(self.data_path, "train", "behaviors.tsv")
        with open(train_behavior_file, "r") as f:
            for line in f.readlines():
                splitted = line.strip("\n").split("\t")
                warm_user.add(splitted[1])
                hist_news = set(splitted[3].split())
                cand_news = set(map(lambda x: x[:-2], splitted[4].split()))
                warm_news = warm_news.union(hist_news)
                warm_news = warm_news.union(cand_news)
        return warm_news, warm_user

    def read_json(self, file_name):
        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as fa:
            file_dict = json.load(fa)
            return file_dict

    def get_news_idx(self, news_id):
        '''
        for warm news, use it idx
        for cold news, all news share a same idx (when testing, because their embedding is not trained, 
        so we use the same embedding)
        '''
        if news_id in self.warm_news:
            return self.news2idx[news_id]
        else:
            return len(self.warm_news)
    
    def get_user_idx(self, user_id):
        if user_id in self.warm_user:
            return self.user2idx[user_id]
        else:
            return len(self.warm_user)

    def read_news(self, news: Dict[str, Any], filepath: str, drop_stopword: bool=True)->Dict[str, Any]:
        '''
        convert news(news_id, category, subcategory, title, category) into numeric data
        '''
        with open(os.path.join(filepath, 'news.tsv'), encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            splitted = line.strip("\n").split("\t")
            news_id = splitted[0]
            if news_id in news:
                continue

            category = splitted[1].lower()
            subcategory = splitted[2].lower()
            title = splitted[3].lower()
            abstract = splitted[4].lower()

            if drop_stopword:
                title = remove_stopword(title)
                abstract = remove_stopword(abstract)
            
            news[news_id] = dict()
            news[news_id]["news_idx"] = self.get_news_idx(news_id)
            news[news_id]["category"] = self.category2id[category]
            news[news_id]["subcategory"] = self.subcategory2idx[subcategory]
            news[news_id]["title"] = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(title))
            news[news_id]["abstract"] = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(abstract))
        return news

    def process_news(self):
        filepath = os.path.join(self.data_path, "news_dict.pkl")
        # if os.path.exists(filepath):
        #     print("Load news info from", filepath)
        #     with open(filepath, 'rb') as fin: news = pickle.load(fin)
        #     return news

        news = dict()
        news = self.read_news(news, os.path.join(self.data_path, 'train'))
        news = self.read_news(news, os.path.join(self.data_path, 'dev'))
        if self._split == 'large':
            news = news.read_news(news, os.path.join(self.data_path, "test"))

        print("Saving news info from", filepath)
        with open(filepath, 'wb') as fout:
            pickle.dump(news, fout)
        return news

    def get_examples(self, negative_sampling=4) -> Any:
        behavior_file = os.path.join(self.data_path, self._mode, "behaviors.tsv")
        df = pd.read_csv(behavior_file, sep='\t', header=None, 
                         names=["impression_id", "user_id", "time", "news_history", "impressions"])
        
        ctr_file = os.path.join(self.data_path, "ctr.csv")
        ctr = pd.read_csv(ctr_file)
        ctr = dict(zip(ctr["news_id"], ctr["ctr"].apply(lambda x: float(x))))

        recency_file = os.path.join(self.data_path, self._mode, "recency.csv")
        recency = pd.read_csv(recency_file)
        df["recency"] = recency["recency"].apply(lambda x: list(map(int, str(x).split(", "))))
        df["recency_dict"] = list(map(lambda x, y: get_recency_dict(x, y), df["impressions"], df["recency"]))

        df["news_history"] = df["news_history"].fillna('')

        if self._mode == "train" and negative_sampling is not None:
            df["impressions"] = df["impressions"].apply(lambda x: sampling(x, ratio=negative_sampling))
        
        if self._mode == "train":
            df = df.drop("impressions", axis=1).join(df['impressions'].str.split('\t', expand=True).stack().
                                                     reset_index(level=1, drop=True).rename('impression'))
            df = df.reset_index(drop=True)
        else:
            df = df.drop('impressions', axis=1).join(df['impressions'].str.split(' ', expand=True).stack().
                                                     reset_index(level=1, drop=True).rename('impression'))
            df = df.reset_index(drop=True)
        
        if self._mode == 'test':
            df['news_id'] = df['impression']
            df['click'] = [-1] * len(df)
        else:
            df['news_id'] = df['impression'].apply(lambda x: process_news_id(x))
            df['click'] = df['impression'].apply(lambda x: process_click(x))
                
        if self._mode == "train":
            df["ctr"] = df["news_id"].apply(lambda x: [ctr[t] for t in x]) 
            df["recency"] = list(map(lambda x, y: get_recency(x, y), df["recency_dict"], df["impression"]))
        else:
            df["ctr"] = df["impression"].apply(lambda x: [ctr[x.split("-")[0]]])
            df["recency"] = list(map(lambda x, y: get_recency(x, y), df["recency_dict"], df["impression"]))
        
        df = df.drop(["time", "recency_dict", "impression"], axis=1)
        df["news_history"] = df["news_history"].apply(lambda x: x.split())

        # generate curr_news/ hist_news/ user idx and cold_mask
        df["user_idx"] = df["user_id"].apply(lambda x: self.get_user_idx(x))
        df["curr_idx"] = df["news_id"].apply(lambda x: [self.get_news_idx(t) for t in x])
        df["hist_idx"] = df["news_history"].apply(lambda x: [self.get_news_idx(t) for t in x])
        df["user_cold_mask"] = df["user_id"].apply(lambda x: (x in self.warm_user))
        df["curr_cold_mask"] = df["news_id"].apply(lambda x: [(t in self.warm_news) for t in x])
        df["hist_cold_mask"] = df["news_history"].apply(lambda x: [(t in self.warm_news) for t in x])
        return df      

    def pack_bert_feature(self, example: Any):

        if self._data_type == 'title':
            curr_input_ids = [self._news[x]['title'] for x in example["news_id"]]
        else:
            curr_input_ids = [self._news[x]["titile"]+self._news[x]["abstract"] for x in example["news_id"]]
        curr_input_ids = [[self._tokenizer.cls_token_id] + x[:self._seq_max_len - 2] + [self._tokenizer.sep_token_id]
                          for x in curr_input_ids]
        curr_category_ids = [self._news[x]["category"]  for x in example["news_id"]]
        curr_token_type = [[0] * len(x) for x in curr_input_ids]
        curr_input_mask = [[1] * len(x) for x in curr_input_ids]
        curr_idx = example["curr_idx"]
        curr_cold_mask = [np.int32(x) for x in example["curr_cold_mask"]]

        curr_padding_len = [self._seq_max_len - len(x) for x in curr_input_ids]
        curr_input_ids = [x + [self._tokenizer.pad_token_id] * curr_padding_len[idx]
                          for idx, x in enumerate(curr_input_ids)]
        curr_token_type = [x + [0] * curr_padding_len[idx] for idx, x in enumerate(curr_token_type)]
        curr_input_mask = [x + [0] * curr_padding_len[idx] for idx, x in enumerate(curr_input_mask)]
        

        hist_news = {
            'hist_input_ids': [],
            'hist_token_type': [],
            'hist_input_mask': [],
            'hist_mask': [],
            'hist_category_ids': [],
        }
        for i, ns in enumerate(example['news_history'][:self._hist_max_len]):
            if self._data_type == 'title':
                hist_input_ids = self._news[ns]['title']
            else:
                hist_input_ids = self._news[ns]['abstract']
            hist_category_ids = self._news[ns]["category"]
            hist_input_ids = hist_input_ids[:self._seq_max_len - 2]
            hist_input_ids = [self._tokenizer.cls_token_id] + hist_input_ids + [self._tokenizer.sep_token_id]
            hist_token_type = [0] * len(hist_input_ids)
            hist_input_mask = [1] * len(hist_input_ids)
            hist_padding_len = self._seq_max_len - len(hist_input_ids)
            hist_input_ids = hist_input_ids + [self._tokenizer.pad_token_id] * hist_padding_len
            hist_token_type = hist_token_type + [0] * hist_padding_len
            hist_input_mask = hist_input_mask + [0] * hist_padding_len
            assert len(hist_input_ids) == len(hist_token_type) == len(hist_input_mask)
            hist_news['hist_input_ids'].append(hist_input_ids)
            hist_news['hist_token_type'].append(hist_token_type)
            hist_news['hist_input_mask'].append(hist_input_mask)
            hist_news['hist_mask'].append(np.int32(1))
            hist_news["hist_category_ids"].append(hist_category_ids)
        hist_idx = [np.int32(x) for x in example["hist_idx"]][:self._hist_max_len]
        hist_cold_mask = [np.int32(x) for x in example["hist_cold_mask"]][:self._hist_max_len]
        
        hist_padding_num = self._hist_max_len - len(hist_news['hist_input_ids'])
        for idx in range(hist_padding_num): 
            hist_news['hist_input_ids'].append([self._tokenizer.pad_token_id] * self._seq_max_len)
            hist_news['hist_token_type'].append([0] * self._seq_max_len)
            hist_news['hist_input_mask'].append([0] * self._seq_max_len)
            hist_news['hist_mask'].append(np.int32(0))
            hist_news['hist_category_ids'].append(np.int32(CATEGORY_CNT))
            hist_idx.append(np.int32(0))
            hist_cold_mask.append(np.int32(0))
        
        user_idx, user_cold_mask = np.int32(example["user_idx"]), np.int32(example["user_cold_mask"])

        return curr_input_ids, curr_token_type, curr_input_mask, curr_category_ids, hist_news, \
            curr_idx, hist_idx, user_idx, curr_cold_mask, hist_cold_mask, user_cold_mask

    def __getitem__(self, index: int) -> Dict[str, int]:
        example = self._examples.iloc[index]
        curr_input_ids, curr_token_type, curr_input_mask, curr_category_ids, hist_news,\
            curr_idx, hist_idx, user_idx, curr_cold_mask, hist_cold_mask, user_cold_mask  = self.pack_bert_feature(example)
        
        ctr, recency = example['ctr'], example['recency']
        
        input = {
            'curr_input_ids': curr_input_ids,
            'curr_token_type': curr_token_type,
            'curr_input_mask': curr_input_mask,
            'curr_category_ids': curr_category_ids,
            'hist_news': hist_news,
            'curr_idx': curr_idx,
            'hist_idx': hist_idx,
            'user_idx': user_idx,
            'curr_cold_mask': curr_cold_mask,
            'hist_cold_mask': hist_cold_mask,
            'user_cold_mask': user_cold_mask,
            'ctr': ctr,
            'recency': recency,
        }
        if self._mode == "train":
            input["click_label"] = example["click"]
            return input
        elif self._mode == "dev":
            input["impression_id"] = example["impression_id"]
            input["click_label"] = example["click"]
            return input
        elif self._mode == "test":
            pass
        else:
            raise ValueError("Mode must be train, dev or test.")

    def __len__(self):
        return len(self._examples)

    def collate_fn(self, batch: Dict[str, Any]):
        curr_input_ids = torch.tensor([item['curr_input_ids'] for item in batch])
        curr_token_type = torch.tensor([item['curr_token_type'] for item in batch])
        curr_input_mask = torch.tensor([item['curr_input_mask'] for item in batch])
        curr_category_ids = torch.tensor([item['curr_category_ids'] for item in batch])
        hist_input_ids = torch.tensor([item['hist_news']['hist_input_ids'] for item in batch])
        hist_token_type = torch.tensor([item['hist_news']['hist_token_type'] for item in batch])
        hist_input_mask = torch.tensor([item['hist_news']['hist_input_mask'] for item in batch])
        hist_mask = torch.tensor([item['hist_news']['hist_mask'] for item in batch])
        hist_category_ids = torch.tensor([item['hist_news']['hist_category_ids'] for item in batch])
        curr_idx = torch.tensor([item['curr_idx'] for item in batch], dtype=torch.int)
        hist_idx = torch.tensor([item['hist_idx'] for item in batch], dtype=torch.int)
        user_idx = torch.tensor([item['user_idx'] for item in batch], dtype=torch.int)
        curr_cold_mask = torch.tensor([item['curr_cold_mask'] for item in batch], dtype=torch.bool)
        hist_cold_mask = torch.tensor([item['hist_cold_mask'] for item in batch], dtype=torch.bool)
        user_cold_mask = torch.tensor([item['user_cold_mask'] for item in batch], dtype=torch.bool)
        ctr = torch.tensor([item['ctr'] for item in batch])
        recency = torch.tensor([item['recency'] for item in batch])

        inputs = {
            'curr_input_ids': curr_input_ids,
            'curr_token_type': curr_token_type,
            'curr_input_mask': curr_input_mask,
            'curr_category_ids': curr_category_ids,
            'hist_input_ids': hist_input_ids,
            'hist_token_type': hist_token_type,
            'hist_input_mask': hist_input_mask,
            'hist_mask': hist_mask,
            'hist_category_ids': hist_category_ids,
            'curr_idx': curr_idx,
            'hist_idx': hist_idx,
            'user_idx': user_idx,
            'curr_cold_mask': curr_cold_mask,
            'hist_cold_mask': hist_cold_mask,
            'user_cold_mask': user_cold_mask,
            'ctr': ctr,
            'recency': recency,
        }

        if self._mode == 'train':
            inputs['click_label'] = torch.tensor([item['click_label'] for item in batch])
            return inputs
        elif self._mode == 'dev':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            inputs['click_label'] = torch.tensor([item['click_label'] for item in batch])
            return inputs
        elif self._mode == 'test':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            return inputs
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')


class MindDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, 
                 pin_memory=False, sampler=None) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            collate_fn=dataset.collate_fn
        )