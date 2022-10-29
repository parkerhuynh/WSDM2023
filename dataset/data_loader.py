import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
import copy
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import operator
import pickle
import argparse
import collections
import logging
import json
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
# from transformers import BertTokenizer,BertModel
from utils.transforms import letterbox, random_affine
import pandas as pd

class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

def processing(df, phase):
    
    """Create a data list to store all raw data simples"""
    data = []
    for idx in range(len(df)):
        img_path, W, H, l, t, r,b, question = df.iloc[idx]
        
        dic  = {
            "question": question
        }
        if phase != "test":
            
            img_path = '/home/ngoc/data/WSDM2023/train_imgs/' + img_path.split('/')[-1]
            dic["img_path"] = img_path
            dic["bb"] = np.array([W, H,l,t,r,b], dtype=float)
        else:
            image_path = '/home/ngoc/data/WSDM2023/test_imgs/' + img_path.split('/')[-1]
            dic["img_path"] = img_path
        data.append(dic)
        
    return data

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class WSDMDataset(data.Dataset):
    def __init__(self, dataset, phase, img_size, max_query_len = 40, transform = None, augment = False, bert_model = 'bert-base-uncased'):
        super().__init__()
        self.dataset = dataset
        self.phase = phase
        self.img_size = img_size
        self.max_query_len = max_query_len
        self.bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        #Visual Processing
        img_path = self.dataset[idx]['img_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.zeros_like(img)
        img, mask, ratio, dw, dh = letterbox(img, mask, self.img_size[0])

        if self.transform:
            img = self.transform(img)
        
        #Bounding Box processing
        bbox = self.dataset[idx]['bb'][2:] #[x_min, y_min, x_max, y_max]
        bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
        bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh

        #Question processing
        question = self.dataset[idx]['question']
        examples = read_examples(question, idx)
        features = convert_examples_to_features(
                examples=examples, seq_length=self.max_query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        return img, mask,  np.array(word_id, dtype=int),  np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)



def data_loaders(data_dir, batch_size, img_size, input_transform,  max_query_len, bert_model):
    train = pd.read_csv(data_dir + 'train.csv')
    df_test = pd.read_csv(data_dir + 'test_public.csv')[:100]
    train_length = int(len(train)*0.8)
    df_train = train[:train_length]
    df_valid = train[train_length:]
    
    df_train = processing(df_train, 'train')
    df_valid = processing(df_valid, 'val')
    df_test = processing(df_test, 'test')

    train_dataset = WSDMDataset(
        df_train, 'train',
        img_size = img_size,
        transform = input_transform,
        bert_model = bert_model,
        max_query_len = max_query_len,
        )
    valid_dataset = WSDMDataset(
        df_train, 'val',
        img_size = img_size,
        transform = input_transform,
        bert_model = bert_model,
        max_query_len = max_query_len,
        )


    train_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size = batch_size,
            shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size = batch_size,
            shuffle=True)
    return {
        'train': train_loader,
        'valid': valid_loader
    }