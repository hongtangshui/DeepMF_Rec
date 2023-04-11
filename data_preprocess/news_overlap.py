'''
Statistic the overlap between train_set hist_new / train_candidates_news / test_hist_news / test_candidates_news
'''
import os
import numpy as np

def get_news(scale):
    train_behaviors_path = os.path.join("data", scale, "train", "behaviors.tsv")
    train_hist_news = set()
    train_cand_news = set()
    train_user = set()
    with open(train_behaviors_path, 'r') as f:
        for line in f.readlines():
            _, user, _, hist_news, candidates_news = line.split('\t')
            train_user.add(user)
            for hist in hist_news.split():
                train_hist_news.add(hist)
            for candidates in candidates_news.split():
                train_cand_news.add(candidates[:-2])
    test_behaviors_path = os.path.join("data", scale, "dev", "behaviors.tsv")
    test_hist_news = set()
    test_cand_news = set()
    test_user = set()
    with open(test_behaviors_path, 'r') as f:
        for line in f.readlines():
            _, user, _, hist_news, candidates_news = line.split('\t')
            test_user.add(user)
            for hist in hist_news.split():
                test_hist_news.add(hist)
            for candidates in candidates_news.split():
                test_cand_news.add(candidates[:-2])

    print(len(train_hist_news), len(train_cand_news), len(test_hist_news), len(test_cand_news))
    print(len(train_hist_news.intersection(train_cand_news)))
    print(len(train_hist_news.intersection(test_hist_news)))
    print(len(test_hist_news.intersection(test_cand_news)))
    print(len(train_cand_news.intersection(test_cand_news)))
    print("--")
    print(len(train_hist_news.union(train_cand_news)))

    cnt = 0
    for news in test_cand_news:
        if news in train_hist_news or news in train_cand_news or news in test_hist_news:
           cnt += 1 
    print(cnt)
    print("-------------user----------------")
    print(len(train_user), len(test_user))
    print(len(train_user.intersection(test_user)))
    
    user_with_warm_news_cnt = 0
    all = 0
    with open(test_behaviors_path, 'r') as f:
        for line in f.readlines():
            _, user, _, hist_news, candidates_news = line.split('\t')
            test_user.add(user)
            flag = 0
            all += 1
            for hist in hist_news.split():
                if hist in train_hist_news or hist in train_cand_news:
                    flag = 1
            user_with_warm_news_cnt += flag
    print(user_with_warm_news_cnt)
    print(all)
    return 0

if __name__ == "__main__":
    get_news("small")