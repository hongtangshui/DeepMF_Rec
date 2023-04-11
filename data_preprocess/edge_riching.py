'''
Add edge in Histnews-User bipartite graph by random walk method.
'''

import pandas as pd
import numpy as np
import os
import json
import random

def HistnewsUserDict(scale, mode):
    '''
    Create a dict for hist_news, whose value is a list of users that clicked it(the news is in the user's history click news) 
    '''
    behaviors_file_path = os.path.join('data', scale, mode, 'behaviors.tsv')
    histnews_user_dict = {}
    user_histnews_dict = {}
    with open(behaviors_file_path, 'r') as f:
        for user_item_data in f.readlines():
            _, user_id, time, hist_news_list, candidate_news_list = user_item_data.strip().split("\t")
            user_histnews_dict[user_id] = hist_news_list.split()
            for hist_news in hist_news_list.split():
                if hist_news not in histnews_user_dict.keys():
                    histnews_user_dict[hist_news] = [user_id]
                else:
                    histnews_user_dict[hist_news].append(user_id)
    # histnews_user_path = os.path.join('data', scale, 'train', "histnews_user.tsv")
    # with open(histnews_user_path, 'w') as f:
    #     for key in histnews_user_dict.keys():
    #         user_list = histnews_user_dict[key]
    #         f.write(key+"\t"+" ".join(user_list)+"\n")
    return user_histnews_dict , histnews_user_dict

def random_walk(scale, mode, extra_num, max_hop=2):
    '''
    For each user, sample extra_num pieces of news to enhance the user's representation by random walk method   
    '''
    user_extra_histnews_dict = {}
    user_histnews_dict , histnews_user_dict = HistnewsUserDict(scale, mode)
    for key in user_histnews_dict.keys():
        extra_histnews_list = {}
        for i in range(extra_num * 20):
            temp_user = key
            if len(user_histnews_dict[temp_user]) == 0:
                break   # no click history, can't use random walk
            temp_news = random.choice(user_histnews_dict[temp_user])
            for j in range(max_hop - 1):
                temp_user = random.choice(histnews_user_dict[temp_news])
                temp_news = random.choice(user_histnews_dict[temp_user])
                if len(user_histnews_dict[temp_user]) == 0:
                    break   # maybe exist other road.
            if temp_news in extra_histnews_list.keys():
                extra_histnews_list[temp_news] += 1
            else:
                extra_histnews_list[temp_news] = 1
        if len(extra_histnews_list) == 0:
            user_extra_histnews_dict[key] = []
            continue
        extra_histnews_list = sorted(extra_histnews_list.items(), key=lambda x: x[1], reverse=True)
        extra_histnews_list = list(zip(*extra_histnews_list[:extra_num]))[0]
        user_extra_histnews_dict[key] = list(extra_histnews_list)
    behaviors_file_path = os.path.join('data', scale, mode, 'behaviors.tsv')
    bahaviors_riched_file_path = os.path.join('data', scale, mode, 'behaviors_riched.tsv')
    histnews_user_dict = {}
    user_histnews_dict = {}
    with open(bahaviors_riched_file_path, 'w') as riched_file:
        with open(behaviors_file_path, 'r') as f:
            for user_item_data in f.readlines():
                line_list = list(user_item_data.split("\t"))
                line_list[3] = " ".join(line_list[3].split()+user_extra_histnews_dict[line_list[1]])
                riched_file.write("\t".join(line_list))





if __name__ == '__main__':
    scale = "small"
    random_walk(scale, "dev", 5, 2)
    random_walk(scale, "train", 5, 2)
    