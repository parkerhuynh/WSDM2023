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
import torch
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
            "question": question,
            "img_path": '/home/ngoc/data/WSDM2023/train_imgs/' + img_path.split('/')[-1],
            "bb": np.array([l,t,r,b], dtype=float)
        }
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
    def __init__(self, dataset, phase, img_size, max_query_len = 40, transform = False, augment = False, bert_model = 'bert-base-uncased', lstm = False, testmode = False):
        super().__init__()
        self.dataset = dataset
        self.phase = phase
        self.imsize = img_size
        self.query_len = max_query_len
        self.bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.transform = transform
        self.augment = augment
        self.lstm = lstm
        self.testmode = testmode

    def pull_item(self, idx):
        img_path = self.dataset[idx]["img_path"]
        bbox = list(self.dataset[idx]["bb"])
        phrase = self.dataset[idx]['question']

        img = cv2.imread(img_path)
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, phrase, bbox

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        #Visual Processing
        phrase= phrase.lower()
        bbox = list(bbox)
        phrase_out = phrase
        if self.augment:
            augment_flip, augment_hsv, augment_affine = True,True,True

        ## seems a bug in torch transformation resize, so separate in advance
        h, w = img.shape[0], img.shape[1]
        #print(h, w)
        #result = img.copy()
        #cv2.rectangle(result, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 5)
        #cv2.imwrite(f'./output/{phrase}_1.jpg', result)
        #mask = copy.deepcopy(img)
        mask = np.zeros_like(img)
        if self.augment:  # True
            ## random horizontal flip
            if augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
                phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            ## random affine transformation
            if augment_affine:
                img, _, bbox, M = random_affine(img, mask, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
        else:   ## should be inference, or specified training
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            #result = img.copy()
            #cv2.rectangle(result, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 5)
            #cv2.imwrite(f'./output/{phrase}_2.jpg', result)


        ## Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)
        if self.lstm:
            phrase_out = self.tokenize_phrase(phrase_out)
            word_id = phrase_out
            # word_mask = np.zeros(word_id.shape)
            word_mask = np.array(word_id>0,dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase_out, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
            word_split = features[0].tokens[1:-1]
        if self.testmode:
            return img, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0], phrase_out, word_split
        else:
            return img, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
            np.array(bbox, dtype=int)



def data_loaders(data_dir, batch_size, img_size, input_transform,  max_query_len, bert_model):
    train = pd.read_csv(data_dir + 'train.csv')
    train_length = int(len(train)*0.8)
    df_train = train[:train_length]
    df_valid = train[train_length:]    
    df_train = processing(df_train, 'train')
    df_valid = processing(df_valid, 'val')

    train_dataset = WSDMDataset(
        df_train, 'train',
        img_size = img_size,
        transform = input_transform,
        bert_model = bert_model,
        max_query_len = max_query_len,
        )
    valid_dataset = WSDMDataset(
        df_valid, 'valid',
        img_size = img_size,
        transform = input_transform,
        bert_model = bert_model,
        max_query_len = max_query_len,
        )


    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size = batch_size)
    valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,)
    return {
        'train': train_loader,
        'valid': valid_loader
    }