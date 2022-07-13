# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import pandas as pd
import argparse
import logging
import random
import numpy as np
from copy import deepcopy
from tblib import Code

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification)
from model import Model
from headandloss import HeadAndLoss
from datasets import TextDataset, DaconSubmitDataset, ClassRandomSampler
from testkit import code1v1verification, batch_cosine_similarity
from utils import set_seed, is_parallel, AverageMeter, select_device, mkdir_if_missing, histogram_plot

logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
import multiprocessing
cpu_cont = 16


def train_per_epoch(args, model, headandloss, train_dataloader, optimizer, scheduler, idx):
    model.train()
    losses = AverageMeter()
    meta_dict = {'spmin' : AverageMeter(), 'snmax' : AverageMeter()}
    
    for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            labels = batch[3].to(args.device)        
          
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            # code_vec = code_vec[:, 0, :] # take <s> token (equiv. to [CLS])
            # code_vec = torch.mean(code_vec, dim=1)
            loss, metas = headandloss(code_vec, labels)

            #report loss
            losses.update(loss.item(), labels.size(0))
            for key, val in metas.items():            
                meta_dict[key].update(val.item(), labels.size(0))
            
            if (step+1)% 10==0:
                s = ''
                for key, val in meta_dict.items():
                    s += key + ' {:.6f} ({:.6f}) '.format(val.val, val.avg)
                logger.info("epoch {} step {}/{} Loss {:.6f} ({:.6f}), {}".format(idx,step+1,len(train_dataloader),losses.val, losses.avg, s))
                
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 



def eval(args, model, tokenizer, pool):
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.code_folder, pool, mode='train')
    valid_dataset=TextDataset(tokenizer, args, args.code_folder, pool, mode='valid')

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = RandomSampler(valid_dataset)
    

    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,num_workers=4)    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.batch_size,num_workers=4)
    
    acc, th = code1v1verification(model, train_dataloader, args.device, 'config/train_pair.txt') 
    logger.info('cosine verification accuracy: {} threshold: {}\n\n'.format(acc, th)) 

    acc, th = code1v1verification(model, valid_dataloader, args.device, 'config/valid_pair_10.txt') 
    logger.info('cosine verification accuracy: {} threshold: {}\n\n'.format(acc, th)) 


def train(args, model, headandloss, tokenizer, pool):
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.code_folder, pool, mode='train')
    valid_dataset=TextDataset(tokenizer, args, args.code_folder, pool, mode='valid')
    valid_sampler = RandomSampler(valid_dataset)
    batch_sampler = ClassRandomSampler(batch_size=args.batch_size,
                                       samples_per_class=2,
                                       image_dict=train_dataset.example_per_class_dict,
                                       image_list=train_dataset.examples)

    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,num_workers=4)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.batch_size,num_workers=4)

    test_file='./test.csv'
    submit_dataset=DaconSubmitDataset(tokenizer, args, test_file, True, pool)    
    submit_dataloader = DataLoader(submit_dataset, batch_size=args.batch_size, num_workers=4)
    
    args.max_steps=args.num_train_epochs*len(train_dataloader)
    args.save_steps=len( train_dataloader)//10
    args.warmup_steps=args.max_steps//10

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    #get optimizer and scheduler
    #optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer.add_param_group({'params': headandloss.parameters()})
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.batch_size)    
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    best_acc, acc, best_th, th = 0.0, 0.0, 0.0, 0.0
    for idx in range(args.num_train_epochs):         
        train_per_epoch(args, model, headandloss, train_dataloader,
                        optimizer, scheduler, idx)
        
        # if (idx + 1) % 1 == 0 or (idx + 1) == args.num_train_epochs:
        acc, th = code1v1verification(model, valid_dataloader, args.device) 
        logger.info('cosine verification accuracy: {} threshold: {}\n\n'.format(acc, th)) 
        # similarities, pair_ids = submit_one_iter(args, model, submit_dataloader)       
        # histogram_plot(similarities, num=100, save_name=os.path.join(args.save_dir, 'test_hist_{}.png'.format(idx+1)))
        
        if acc >= best_acc:
            best_acc = acc  
            best_th = th
        ckpt = {'epoch' : idx,                     
                'best_acc': best_acc,
                'best_th': best_th,                     
                'backbone' : deepcopy(model.module if is_parallel(model) else model).eval(),
                }

        torch.save(ckpt, args.save_dir + '/last.pt')
        if best_acc == acc:
            torch.save(ckpt, args.save_dir + '/best.pt')
        del ckpt

    similarities, pair_ids = submit_one_iter(args, model, submit_dataloader)       
    histogram_plot(similarities, num=100, save_name=os.path.join(args.save_dir, 'submit_valid_hist.png'))

def submit_one_iter(args, model, submit_dataloader):
    model.eval()
    similarities = []
    pair_ids = []
    with torch.no_grad():
        for step,batch in enumerate(submit_dataloader):
            #get inputs
            code_inputs_1 = batch[0].to(args.device)  
            attn_mask_1 = batch[1].to(args.device)
            position_idx_1 = batch[2].to(args.device)

            code_inputs_2 = batch[3].to(args.device)  
            attn_mask_2 = batch[4].to(args.device)
            position_idx_2 = batch[5].to(args.device)
                        
            pair_id = batch[6]                        
            pair_ids.append(pair_id.tolist())

            #get code and nl vectors
            code_vec_1 = model(code_inputs=code_inputs_1,attn_mask=attn_mask_1,position_idx=position_idx_1)                        
            # code_vec_1 = code_vec_1[:, 0, :] # take <s> token (equiv. to [CLS])

            code_vec_2 = model(code_inputs=code_inputs_2,attn_mask=attn_mask_2,position_idx=position_idx_2)                        
            # code_vec_2 = code_vec_2[:, 0, :] # take <s> token (equiv. to [CLS])

            sim = batch_cosine_similarity(code_vec_1, code_vec_2)
            similarities.extend(sim.data.cpu().tolist())
            
            if (step+1)% 10==0:                
                logger.info("step {}/{}".format(step+1,len(submit_dataloader)))
    return similarities, pair_ids


def submit(args, model, tokenizer, pool):       
    test_file='./test.csv'
    submit_dataset=DaconSubmitDataset(tokenizer, args, test_file, False, pool)    
    submit_dataloader = DataLoader(submit_dataset, batch_size=args.batch_size, num_workers=4)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)    
    similarities, pair_ids = submit_one_iter(args, model, submit_dataloader)
        
    histogram_plot(similarities, num=100, save_name=os.path.join(args.save_dir, 'hist_submit2.png'))

    submission = pd.read_csv('./sample_submission.csv')
    submission['similar'] = similarities            
    submission.to_csv(os.path.join(args.save_dir, 'submission_raw2.csv'), index=False)

    # th = 0.0819
    # similarities = (np.array(similarities) >= th).astype(int)
    # submission['similar'] = similarities
    # submission.to_csv(os.path.join(args.save_dir, 'submission{}.csv'.format(th)), index=False)


def arg_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--code_folder", default=None, type=str,
                        help="The input training data file (a json file).")    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_submit", action='store_true',
                        help="Whether to run eval on the test set.")  
    

    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size for training.")    
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--head', type=str, default='arcface', help='e.g., arcface, cosface, magface')    
    parser.add_argument('--aux', type=str, default='', help='unified negative pair generation (e.g., unpg, unpgfw)')    
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs_eccv_rebuttal', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    parser.add_argument('--reverse_tokens', action='store_true', help='')
    #print arguments    
    args = parser.parse_args()

    return args


def main():
    args = arg_parser()    
    args.save_dir = os.path.join(args.project, args.model_name_or_path.split('/')[-1] + '.' + args.head + '.' + args.aux + '.' + args.name)
    mkdir_if_missing(args.save_dir)
    print(args.save_dir)
    
    pool = multiprocessing.Pool(cpu_cont)

    #set log
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(args.save_dir, 'train.log'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    #set device
    device = select_device(args.device, batch_size=args.batch_size)    
    args.device = device
    args.n_gpu = torch.cuda.device_count()    
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # model = RobertaModel.from_pretrained(args.model_name_or_path)    
    # model = Model(model)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    headandloss = HeadAndLoss(in_feature=768, 
                              num_classes=300, 
                              head_name=args.head, 
                              aux_name=args.aux).to(args.device)
    # Training
    if args.do_train:
        train(args, model, headandloss, tokenizer, pool)

    if args.do_submit:
        # model_path = os.path.join(args.save_dir, 'best.pt')
        model_path = os.path.join(args.save_dir, 'last.pt')
        ckpt = torch.load(model_path, map_location=device)  # load checkpoint        
        model = ckpt['backbone'].to(args.device)
        submit(args, model, tokenizer, pool)
    
    if args.do_eval:
        model_path = os.path.join(args.save_dir, 'best.pt')
        ckpt = torch.load(model_path, map_location=device)  # load checkpoint        
        model = ckpt['backbone'].to(args.device)
        eval(args, model, tokenizer, pool)
    

if __name__ == "__main__":
    main()