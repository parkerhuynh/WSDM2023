
import  numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from utils import make_vocab_questions, tokenize, Text_Dict, bb_processing
from config import data_cfg


def vqa_processing(df, phase):
    
    """Create a data list to store all raw data simples"""
    data = []
    for idx in range(len(df)):
        image_path, W, H, l, t, r,b, question = df.iloc[idx]
        question_token = tokenize(question)
        dic  = {
            "image_path" : image_path,
            "question_text": question,
            "question_token": question_token,
        }
        if phase != "test":
            dic["actual_bounding_box"] = [W, H,l,t,r,b]
            w = r - l
            h = b - t
            dic["bounding_box"] = bb_processing.bb_normalise(l,t,w,h, W, H)
        data.append(dic)
        
    return data

class VqaDataset(data.Dataset):

    """Processing each data batch"""

    def __init__(self, data, question_vocab, phase, max_question_length = 30, transform = False):
        self.data = data
        self.question_vocab = question_vocab 
        self.max_question_length = max_question_length
        self.transform = transform
        self.question_dict = Text_Dict(self.question_vocab)
        self.phase = phase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data
        max_question_length = self.max_question_length
        transform = self.transform
        question_dict = self.question_dict
        image_path = data[idx]['image_path']
        image = Image.open(image_path).convert('RGB')
        question_2_idx = np.array([question_dict.word2idx('<pad>')] * max_question_length)  # padded with '<pad>' in 'ans_vocab'
        question_2_idx[:len(data[idx]['question_token'])] = [question_dict.word2idx(w) for w in data[idx]['question_token']]
        sample = {'image': image, 'question': question_2_idx}
        if self.phase != "test":
            sample["actual_bounding_box"]  = np.array(data[idx]["actual_bounding_box"], dtype='f')
            sample["bounding_box"]  = np.array(data[idx]["bounding_box"], dtype='f')
        sample['image'] = transform(sample['image'])
        sample['image_path'] = image_path
        sample["question_text"] = data[idx]['question_text']
        return sample
        


def data_loaders(data_dir, max_question_length,batch_size, image_size, test_running):
    
    """Pytorch Data Loader"""

    train = pd.read_csv(data_dir + f'train_{data_cfg["image_size"][0]}.csv')
    #train.image  = train.image.apply(lambda x: data_dir + 'train_imgs/' + str(x.split("/")[-1]))
    #df_test = pd.read_csv(data_dir + 'test_public.csv')
    #df_test.image  = df_test.image.apply(lambda x: data_dir + 'test_imgs/' + str(x.split("/")[-1]))
    
    length = int(len(train)*0.8)
    df_train = train[:length]
    df_val = train[length:]
    df_train = df_train.reset_index(drop =True)
    df_val = df_val.reset_index(drop =True)
    if test_running:
        df_train = df_train[:100]
        df_val = df_val[:100]
        #df_test = df_test[:32]
    print(df_train)
    print(f"the len of the trainnig set: {len(df_train)}")
    print(f"the len of the validation set: {len(df_val)}")
    #print(f"the len of the test set: {len(df_test)}")
    #Create vocab size
    question_vocab = make_vocab_questions(df_train["question"])
    #Create a data list to store all raw data simples
    train_data = vqa_processing(df_train, "train")
    val_data = vqa_processing(df_val, "val")
    #test_data = vqa_processing(df_test, "test")
    
    #Tranform image vectors to Tensor, Resize and Normalize them
    transform = transforms.Compose([transforms.ToTensor(), \
        #transforms.Resize(size = image_size),
        transforms.Normalize((0.485, 0.456, 0.406),\
        (0.229, 0.224, 0.225))])\

    #Create Datasets
    print("- Creating Datasets")
    train_dataset = VqaDataset(
            data = train_data,
            question_vocab = question_vocab,
            max_question_length = max_question_length,
            transform = transform,
            phase ="train")
    
    val_dataset =  VqaDataset(
            data = val_data,
            max_question_length = max_question_length,
            question_vocab = question_vocab,
            transform = transform,
            phase = "val")
    """
    test_dataset =  VqaDataset(
            data = test_data,
            max_question_length = max_question_length,
            question_vocab = question_vocab,
            transform = transform,
            phase = "test")
    """
    #Create Data Loaders
    print("- Creating Data Loaders")
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True)
    """
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True)
    """
    print("- Done!")
    return {
        "train": train_loader,
        'val': val_loader,
        #'test': test_loader
    }