import os
import numpy as np
import pandas as pd


def cal_ctr(scale):
    train_file = os.path.join("data", scale, "train", "behaviors.tsv")
    imp_cnt = {}
    click_cnt = {}
    with open(train_file, "r") as f:
        for line in f.readlines():
            line = line.split("\t")[4]
            for news in line.split():
                news, click = news[:-2], int(news[-1])
                if news in imp_cnt.keys():
                    imp_cnt[news] += 1
                    click_cnt[news] += click                
                else:
                    imp_cnt[news] = 1
                    click_cnt[news] = click
    train_ctr = {}
    for key in imp_cnt.keys():
        train_ctr[key] = (click_cnt[key]+1) / (imp_cnt[key]+20)  # posterior   +11/+288 ?
    print(np.mean(list(click_cnt.values())))
    print(np.mean(list(imp_cnt.values())))
    test_file = os.path.join("data", scale, "dev", "behaviors.tsv")
    dev_ctr = {}
    with open(test_file, "r") as f:
        for line in f.readlines():
            line = line.split("\t")[4]
            for news in line.split():
                news, click = news[:-2], int(news[-1])
                if news in train_ctr.keys():
                    dev_ctr[news] = train_ctr[news]
                else:
                    train_ctr[news] = -1
                    dev_ctr[news] = -1
    news_id, ctr = train_ctr.keys(), train_ctr.values()
    ctr = {"news_id": list(news_id), "ctr":list(ctr)}
    train_ctr = pd.DataFrame(ctr)
    path = os.path.join("data", scale, "masked_ctr.csv")
    train_ctr.to_csv(path)
    news_id, ctr = dev_ctr.keys(), dev_ctr.values()
    ctr = {"news_id": list(news_id), "ctr":list(ctr)}
    dev_ctr = pd.DataFrame(ctr)
    path = os.path.join("data", scale, "masked_dev_ctr.csv")
    dev_ctr.to_csv(path)

if __name__ == "__main__":
    cal_ctr("small")