import numpy
import os
import pickle
import json

def user2idx(split):
    user = []
    news = []
    behavior_file = os.path.join("data", split, "train", "behaviors.tsv")
    with open(behavior_file, 'r') as f:
        for line in f.readlines():
            splitted = line.strip("\n").split("\t")
            user_id = splitted[1]
            hist = splitted[3].split()
            cand = list(map(lambda x: x[:-2], splitted[4].split()))
            if user_id not in user:
                user.append(user_id)
            for n in hist:
                if n not in news:
                    news.append(n)
            for n in cand:
                if n not in news:
                    news.append(n)
    print("-------------------------")
    behavior_file = os.path.join("data", split, "dev", "behaviors.tsv")
    with open(behavior_file, 'r') as f:
        for line in f.readlines():
            splitted = line.strip("\n").split("\t")
            user_id = splitted[1]
            hist = splitted[3].split()
            cand = list(map(lambda x: x[:-2], splitted[4].split()))
            if user_id not in user:
                user.append(user_id)
            for n in hist:
                if n not in news:
                    news.append(n)
            for n in cand:
                if n not in news:
                    news.append(n)
    print(user)
    news_dict = dict(zip(news, range(len(news))))
    user_dict = dict(zip(user, range(len(user))))
    with open(os.path.join("data", split, "news2idx.json"), 'w') as f:
        json.dump(news_dict, f)
    with open(os.path.join("data", split, "user2idx.json"), "w") as f:
        json.dump(user_dict, f)
    return 0

if __name__ == "__main__":
    user2idx("small")
