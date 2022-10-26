import re
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from config import model_cfg

def make_vocab_questions(questions):
    
    """Make dictionary for questions and save them into text file."""

    print("- Creating a vocab list for questions")
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = []

    set_question_length = [None]*len(questions)
    for iquestion, question in enumerate(questions):
        words = SENTENCE_SPLIT_REGEX.split(question.lower())
        words = [w.strip() for w in words if len(w.strip()) > 0]
        vocab_set.update(words)
        set_question_length[iquestion] = len(words)
    question_length += set_question_length

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')
    print(f'    + The size of Question vocabbulary {len(vocab_list)}.')
    print(f'    + Maximum length of question: {np.max(question_length)}')
    return vocab_list

def tokenize(sentence):

    """Split string lines into lists"""
    
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

class Text_Dict:

    """Aim to convert index to word or word to index"""

    def __init__(self, vocab):
        
        self.word_list = vocab
        self.word2idx_dict = {w:n_w for n_w, w in enumerate(self.word_list)}
        self.vocab_size = len(self.word_list)
        self.unk2idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None

    def idx2word(self, n_w):

        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.unk2idx is not None:
            return self.unk2idx
        else:
            raise ValueError(f'word {w} not in dictionary (while dictionary does not contain <unk>)')

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]

        return inds

def bb_convertor(output, image_shape):
    x_center, y_center, w_hat, h_hat = output
    W, H = image_shape
    x_center = x_center.item() *W
    y_center = y_center.item() *H
    h_hat = h_hat.item() *H
    w_hat = w_hat.item() *W
    return [int(x_center - w_hat// 2), int(y_center - h_hat // 2), int(x_center + w_hat // 2), int(y_center + h_hat // 2)]

class bb_processing:
    def __init__(self) -> None:
        pass
    def bb_normalise(l, t, w, h, W, H):
        x_center = (l + (w/2))/W
        y_center = (t + (h/2))/H
        w_hat = w/ W
        h_hat = h/ H
        return [x_center, y_center, w_hat, h_hat]

    def get_bb_predictions(outputs, image_shapes):
        actual_outputs = []
        for i in range(outputs.shape[0]):
            actual_output = bb_convertor(outputs[i], image_shapes[i])
            actual_outputs.append(actual_output)
        return actual_outputs
            
def progress_bar(current, total, data_name, bar_length=30):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n ' if current == total else '\r'

    print(f'    + Progress on {data_name.upper()} SET : [{arrow}{padding}] {int(fraction*100)}%    ', end=ending)

def draw_prediction(img_path, gt_bb1, pred_bb1):
    try:
        pred_bb = []
        gt_bb = []
        for i in range(4):
            pred_bb.append(int(pred_bb1[i]))
            gt_bb.append(int(gt_bb1[i].item()))
        img = cv2.imread(img_path)

        result = img.copy()
        cv2.rectangle(result, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]), (0, 255, 0), 2)
        cv2.rectangle(result, (pred_bb[0], pred_bb[1]), (pred_bb[2], pred_bb[3]), (255, 0, 0), 2)

        cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        image_name = img_path.split("/")[-1]
        img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        img.save(f"{model_cfg['saved_image']}{image_name}")
        return 1
    except:
        return 0
        