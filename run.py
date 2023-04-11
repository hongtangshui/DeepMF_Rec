

import random 
import warnings 
import numpy.random
import torch
import argparse
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, BertTokenizer
from torch.cuda import amp
from tqdm import tqdm
from time import gmtime, strftime

from src.data_loader.data_loader import MindDataset, MindDataLoader
from src.model.model import DeepMatrixFactorization
from src.loss.loss import Loss
from src.evaluation.eval import dev

def set_environment(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default=2022)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--split", type=str, default="small")
    parser.add_argument("--pretrain", type=str, default="pretrainedModel/BERT-base-uncased")
    parser.add_argument("--news_type", type=str, default="title")
    parser.add_argument("--news_mode", type=str, default="cls")
    parser.add_argument('--hist_max_len', type=int, default=50)
    parser.add_argument('--seq_max_len', type=int, default=20)
    parser.add_argument('--score_type', type=str, default='weighted')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--epoch', type=int, default=5) # always set 5 in small dataset, 2 in large
    parser.add_argument('--batch_size', type=int, default=128) # default=128, two 3090 gpus
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--log_name', type=str, default='dmf')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--distribution', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    return args

def restore(model, args):
    pass

WARM_NEWS = 51282
WARM_USER = 50000

def main():
    args = parse_args()
    args.nprocs = torch.cuda.device_count()
    os.makedirs(args.output, exist_ok=True)
    set_environment(args.seed)

    log_file = os.path.join(args.output, "{}-{}-{}-{}.log".format(
        args.log_name, args.mode, args.split, strftime("%Y%m%d%H%M%S", gmtime())
    ))

    def printzzz(log):
        with open(log_file, "a") as fout:
            fout.write(log + "\n")
        print(log)
    
    printzzz(str(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # amp
    enable_amp = True if 'cuda' in device.type else False
    if args.use_amp and enable_amp:
        scaler = amp.GradScaler(enabled=enable_amp)
    
    model = DeepMatrixFactorization(768, WARM_NEWS, WARM_USER, pretrained=args.pretrain, device=device, args=args)
    if args.restore is not None and os.path.isfile(args.restore):
        restore(model, args)
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain)

    if args.mode == "train":
        printzzz('reading training data ...')
        train_set = MindDataset(args.root, tokenizer=tokenizer, mode='train', split=args.split,
                                hist_max_len=args.hist_max_len, seq_max_len=args.seq_max_len, data_type=args.news_type)
        train_loader = MindDataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        
        printzzz('reading dev data')
        dev_set = MindDataset(args.root, tokenizer=tokenizer, mode='dev', split=args.split,
                                hist_max_len=args.hist_max_len, seq_max_len=args.seq_max_len, data_type=args.news_type)
        dev_loader = MindDataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss().to(device)
        loss_calculator = Loss(loss_fn)

        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        m_optim.zero_grad()
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=len(train_set)//args.batch_size*2,
                                                       num_training_steps=len(train_set)*args.epoch//args.batch_size)

        model.zero_grad()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            loss_calculator = nn.DataParallel(loss_calculator)
        
        printzzz("start training ...")

        for epoch in range(args.epoch):
            avg_loss = 0.0
            
            batch_itertor = tqdm(train_loader)
            for step, train_batch in enumerate(batch_itertor):
                if args.use_amp:
                    with amp.autocast(enabled=enable_amp):
                        hist_score, batch_score = model(
                            train_batch['curr_input_ids'].to(device),
                            train_batch['curr_token_type'].to(device),
                            train_batch['curr_input_mask'].to(device),
                            train_batch['curr_category_ids'].to(device),
                            train_batch['hist_input_ids'].to(device),
                            train_batch['hist_token_type'].to(device),
                            train_batch['hist_input_mask'].to(device),
                            train_batch['hist_mask'].to(device),
                            train_batch['hist_category_ids'].to(device),
                            train_batch['curr_idx'].to(device),
                            train_batch['hist_idx'].to(device),
                            train_batch['user_idx'].to(device),
                            train_batch['curr_cold_mask'].to(device),
                            train_batch['hist_cold_mask'].to(device),
                            train_batch['user_cold_mask'].to(device),
                            train_batch['ctr'].to(device),
                            train_batch['recency'].to(device),
                        )
                        batch_loss = loss_calculator(batch_score, hist_score, train_batch['click_label']
                                                     .to(device).float().squeeze(), train_batch['hist_mask'].to(device))
                else:
                    hist_score, batch_score = model(
                            train_batch['curr_input_ids'].to(device),
                            train_batch['curr_token_type'].to(device),
                            train_batch['curr_input_mask'].to(device),
                            train_batch['curr_category_ids'].to(device),
                            train_batch['hist_input_ids'].to(device),
                            train_batch['hist_token_type'].to(device),
                            train_batch['hist_input_mask'].to(device),
                            train_batch['hist_mask'].to(device),
                            train_batch['hist_category_ids'].to(device),
                            train_batch['curr_idx'].to(device),
                            train_batch['hist_idx'].to(device),
                            train_batch['user_idx'].to(device),
                            train_batch['curr_cold_mask'].to(device),
                            train_batch['hist_cold_mask'].to(device),
                            train_batch['user_cold_mask'].to(device),
                            train_batch['ctr'].to(device),
                            train_batch['recency'].to(device),
                        )
                    batch_loss = loss_calculator(batch_score, hist_score, train_batch['click_label']
                                                     .to(device).float().squeeze(), train_batch['hist_mask'].to(device))

                if torch.cuda.device_count() > 1:
                    batch_loss = batch_loss.mean()
                avg_loss += batch_loss.item()

                if args.use_amp:
                    scaler.scale(batch_loss).backward()
                    scaler.step(m_optim)
                    scaler.update()
                else:
                    batch_loss.backward()
                    m_optim.step()
                
                m_scheduler.step()
                m_optim.zero_grad()
            
            # if epoch < 2:
            #     printzzz("Epoch {}, Avg_loss: {:.4f}".format(epoch+1, avg_loss))
            #     continue

            if args.eval:
                if args.use_amp:
                    with amp.autocast(enabled=enable_amp):
                        auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, args.output, epoch)
                else:
                    auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, args.output, epoch)
                printzzz("Epoch {}, Avg_loss: {:.4f}, AUC: {:.4f}, MRR: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4f}"
                         .format(epoch+1, avg_loss, auc, mrr, ndcg5, ndcg10))
            else:
                printzzz("Epoch {}, Avg_loss: {:.4f}".format(epoch+1, avg_loss))

            # final_path = os.path.join(args.output, "epoch_{}.bin".format(epoch+1))
            # if args.save:
            #     if torch.cuda.device_count() > 1:
            #         torch.save(model.module.state_dict(), final_path)
            #     else:
            #         torch.save(model.state_dict(), final_path)

    
if __name__ == "__main__":
    main()
