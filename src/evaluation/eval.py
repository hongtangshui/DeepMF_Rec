
from typing import List, Tuple, Dict, AnyStr

import os
import torch
import multiprocessing
from multiprocessing import Pool
import operator
import functools

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd

def flatten(lists: List[List]) -> List:
    return functools.reduce(operator.iconcat, lists, [])


def auc_func(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    return roc_auc_score(grouped_df["label"], grouped_df["score"])


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(df_groups, k=10):
    y_true = np.array(df_groups['label'])
    y_score = np.array(df_groups['score'])
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 1.0
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(df_groups):
    y_true = np.array(df_groups['label'])
    y_score = np.array(df_groups['score'])
    if np.sum(y_true) == 0:
        return 1.0
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def calculate_single_user_metric(df_groups):
    try:
        auc = auc_func(df_groups)
        mrr = mrr_score(df_groups)
        ndcg5 = ndcg_score(df_groups, 5)
        ndcg10 = ndcg_score(df_groups, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4

def dev(model, dev_loader, device, out_path, epoch=0):
    impression_ids = []
    labels = []
    scores = []

    batch_iterator = tqdm(dev_loader, disable=False)
    for step, dev_batch in enumerate(batch_iterator):
        impression_id, click_label = dev_batch['impression_id'], dev_batch['click_label']
        with torch.no_grad():
            poly_attn, batch_score = model(
                dev_batch['curr_input_ids'].to(device),
                dev_batch['curr_token_type'].to(device),
                dev_batch['curr_input_mask'].to(device),
                dev_batch['curr_category_ids'].to(device),
                dev_batch['hist_input_ids'].to(device),
                dev_batch['hist_token_type'].to(device),
                dev_batch['hist_input_mask'].to(device),
                dev_batch['hist_mask'].to(device),
                dev_batch['hist_category_ids'].to(device),
                dev_batch['curr_idx'].to(device),
                dev_batch['hist_idx'].to(device),
                dev_batch['user_idx'].to(device),
                dev_batch['curr_cold_mask'].to(device),
                dev_batch['hist_cold_mask'].to(device),
                dev_batch['user_cold_mask'].to(device),
                dev_batch['ctr'].to(device),
                dev_batch['recency'].to(device),
            )
            batch_score = batch_score.sigmoid()
            batch_score = batch_score.detach().cpu().tolist()
            click_label = click_label.tolist()
            impression_ids.extend(impression_id)
            labels.extend(click_label)
            scores.extend(batch_score)
    
    labels = flatten(labels)
    scores = flatten(scores)

    score_path = os.path.join(out_path, "dev_score_{}.tsv".format(str(epoch)))
    eval_df = pd.DataFrame()
    eval_df['impression_id'] = impression_ids
    eval_df['label'] = labels
    eval_df['score'] = scores
    eval_df.to_csv(score_path, sep='\t', index=False)

    groups_iter = eval_df.groupby('impression_id')
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    results = pool.map(calculate_single_user_metric, df_groups)
    pool.close()
    pool.join()

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)

