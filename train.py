from loss import generalized_iou_loss as IOU_LOSS, Reg_Loss, RMSELoss
from torch.optim import lr_scheduler
import torch.optim as optim
from config import model_cfg, data_cfg
from metric import IOU
from utils import bb_processing, draw_prediction, progress_bar
import torch.nn as nn
import torch
import wandb
import os
import numpy as np
import matplotlib.image as mpimg
import torch.nn.functional as F



def train(model, train_loader, val_loader, device):
    if model_cfg["pre_train"]:
        model = torch.load(f'{model_cfg["checkpoint_dir"]}{model_cfg["model_name"]}.pt')
    try:
        os.system(f'mkdir {model_cfg["checkpoint_dir"]}')
    except:
        pass
    try:
        os.system(f'mkdir {model_cfg["saved_image"]}')
    except:
        pass

    if model_cfg['wandb_log']:
        wandb.init(project="WSDM2023",
                entity="ngocdunghuynh",
                config= model_cfg)
        wandb.config.update(data_cfg)
        wandb.config.update({"device": device})
        files = ["config.py", "data_loaders.py", "loss.py", "main.py", "metric.py", "train.py", "utils.py", "models/attention.py", "models/language_extraction.py", "models/visual_extraction", "models/models.py"]
        for file in files:
            wandb.save(file)
        columns=["Epoch", "Question", "answer"]
        wandb_table = wandb.Table(columns=columns)
    train_iter = 0
    val_iter = 0
    min_val_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=model_cfg["learning_rate"])
    scheduler = lr_scheduler.StepLR(optimizer, \
        step_size=model_cfg['decay_step_size'],
         gamma=model_cfg["gamma"])
    loss_function = RMSELoss()
    
    for epoch in range(model_cfg["num_epochs"]):
        print(f'- EPOCH {epoch + 1}/{model_cfg["num_epochs"]}')
        train_running_loss = 0.0
        train_running_iou = 0.0
        train_step_size = len(train_loader.dataset) / data_cfg["batch_size"]
        model.train(True)
        scheduler.step()
        
        for i, train_batch_sample in enumerate(train_loader):
            progress_bar(i, train_step_size-1, "train", bar_length=30)
            train_images = train_batch_sample['image'].to(device)
            train_questions = train_batch_sample['question'].to(device)
            train_labels = train_batch_sample['bounding_box'].to(device)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                train_outputs = model(train_images, train_questions)
                

                train_actual_labels = train_batch_sample['actual_bounding_box'][:,2:].to(device)
                train_image_shapes = train_batch_sample['actual_bounding_box'][:,:2]
                train_actual_outputs = torch.Tensor(bb_processing.get_bb_predictions(train_outputs, train_image_shapes)).to(device) ###
                loss =  loss_function(train_labels, train_outputs)
                train_iou = IOU(train_actual_labels, train_actual_outputs)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()
                train_running_iou += train_iou
                train_iter  += 1
                if model_cfg['wandb_log']:
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_iou": train_iou,
                        "train_iter": train_iter
                    })
        

        train_running_loss = train_running_loss/train_step_size
        train_running_iou = train_running_iou/train_step_size
        if model_cfg['wandb_log']:
            wandb.log({
                "epoch_loss": train_running_loss,
                "epoch_iou": train_running_iou,
                "epoch": epoch + 1
            })
        
        model.train(False)
        val_step_size = len(val_loader.dataset) / data_cfg["batch_size"]
        val_running_loss = 0.0
        val_running_iou = 0.0

        for i, val_batch_sample in enumerate(val_loader):
            progress_bar(i, val_step_size-1, "val", bar_length=30)
            val_images = val_batch_sample['image'].to(device)
            val_question_texts =  val_batch_sample["question_text"]
            val_image_paths = val_batch_sample["image_path"]
            val_questions = val_batch_sample['question'].to(device)
            val_labels = val_batch_sample['bounding_box'].to(device)
            with torch.set_grad_enabled(False):
                val_outputs = model(val_images, val_questions)

                val_actual_labels = val_batch_sample['actual_bounding_box'][:,2:].to(device)
                val_image_shapes = val_batch_sample['actual_bounding_box'][:,:2]
                val_actual_outputs = torch.Tensor(bb_processing.get_bb_predictions(val_outputs, val_image_shapes)).to(device)
                val_loss =  loss_function(val_labels,  val_outputs)
                val_iou = IOU(val_actual_labels, val_actual_outputs)
                val_running_loss += val_loss.item()
                val_running_iou += val_iou
                val_iter += 1
                if model_cfg['wandb_log']:
                    wandb.log({
                        "val_loss": val_loss.item(),
                        "val_iou": val_iou,
                        "val_iter": val_iter
                    })
        val_running_loss = val_running_loss/val_step_size
        val_running_iou = val_running_iou/val_step_size
        if model_cfg['wandb_log']:
            wandb.log({
                "val_epoch": epoch+1,
                "epoch_val_loss": val_running_loss,
                "epoch_val_iou": val_running_iou
            })
            indexes = [1,2]
            for id in indexes:
                image_path = val_image_paths[id]
                actual_label = val_actual_labels[id]
                actual_output = val_actual_outputs[id]
                question_text = val_question_texts[id] 
                image_predict = draw_prediction(image_path, actual_label, actual_output)
                

                if image_predict == 0:
                    image_predict = np.nan
                else:
                    image_name = image_path.split("/")[-1]
                    image_predict = wandb.Image(mpimg.imread(f"{model_cfg['saved_image']}/{image_name}"))
                wandb_table.add_data(epoch+1, question_text, image_predict)

        if val_running_loss < min_val_loss:
            min_val_loss = val_running_loss
            torch.save(model, f'{model_cfg["checkpoint_dir"]}/{model_cfg["model_name"]}.pt')
        model.train(True)
    if model_cfg['wandb_log']:
        wandb.log({"Image_Table": wandb_table})
        wandb.save(f'{model_cfg["checkpoint_dir"]}/{model_cfg["model_name"]}.pt')
        wandb.finish()