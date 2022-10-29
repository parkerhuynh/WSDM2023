import os
from utils.transforms import ResizeImage, ResizeAnnotation
import sys
import argparse
import time
import random
import json
import math
from distutils.version import LooseVersion
import scipy.misc
import logging
import datetime
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from dataset.data_loader import data_loaders

from utils.checkpoint import save_checkpoint, load_pretrain, load_resume
import wandb


from models.model import TransVG
from models.loss import Reg_Loss, GIoU_Loss
#from utils.parsing_metrics import *
from utils.utils import *
def main():
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--gpu', default='0,1', help='gpu id')
    parser.add_argument('--workers', default=8, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=256, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_dec', default=0.1, type=float, help='decline of learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--size', default=640, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default='/home/ngoc/data/WSDM2023/',
                        help='path to dataset splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--time', default=40, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',  #
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=5000, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='TransVG_6.3', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--w_div', default=0.125, type=float, help='weight of the diverge loss')
    parser.add_argument('--tunebert', dest='tunebert', default=True, action='store_true', help='if tunebert')
    
    # * DETR
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400+40+1, type=int,
                        help="Number of query slots in VLFusion")
    parser.add_argument('--pre_norm', action='store_true')


    global args, anchors_full
    args = parser.parse_args()
    
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    wandb.init(project="WSDM2023",
                entity="ngocdunghuynh",
                config= args)
    columns=['epoch',"gt_box", "pred_bb"]
    wandb_table = wandb.Table(columns=columns)

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    dataloaders =  data_loaders(
        data_dir=args.data_root,
        batch_size=args.batch_size,
        img_size=(args.size, args.size),
        input_transform=input_transform,
        max_query_len=40,
        bert_model=args.bert_model
    )
    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]
    
    model = TransVG(jemb_drop_out=0.1, bert_model=args.bert_model,tunebert=args.tunebert, args=args)
    model = torch.nn.DataParallel(model).cuda()
    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))

    if args.tunebert:
        visu_param = model.module.visumodel.parameters()
        text_param = model.module.textmodel.parameters()
        rest_param = [param for param in model.parameters() if ((param not in visu_param) and (param not in text_param))]
        visu_param = list(model.module.visumodel.parameters())
        text_param = list(model.module.textmodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in text_param])
        sum_fusion = sum([param.nelement() for param in rest_param])
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)
    else:
        visu_param = model.module.visumodel.parameters()
        rest_param = [param for param in model.parameters() if param not in visu_param]
        visu_param = list(model.module.visumodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
        sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    if args.tunebert:
        optimizer = torch.optim.AdamW([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.},
                {'params': text_param, 'lr': args.lr/10.}], lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = torch.optim.AdamW([{'params': rest_param},
                {'params': visu_param}],lr=args.lr, weight_decay=0.0001)

     ## training
    train_iter = 0
    val_max_iou = -float('Inf')
    for epoch in range(args.nb_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        out_iter = train_epoch(train_loader, model, optimizer, epoch, train_iter)
        train_iter = out_iter
        iou_new = validate_epoch(valid_loader, model,epoch, wandb_table)
        if iou_new > val_max_iou:
            val_max_iou = iou_new
            torch.save(model, 'saved_models/model.pt')
    
    wandb.log({"Image_Table": wandb_table})
    wandb.save('saved_models/model.pt')
    wandb.finish()

    
def train_epoch(train_loader, model, optimizer, epoch, train_iter):
    losses = AverageMeter()
    l1_losses = AverageMeter()
    GIoU_losses = AverageMeter()
    # div_losses = AverageMeter()
    acc = AverageMeter()
    # acc_center = AverageMeter()
    miou = AverageMeter()
    
    model.train()

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        train_iter  += 1
        # print('get data from train_loader...')
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:,:,:,0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        gt_bbox = gt_bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        gt_bbox = Variable(gt_bbox)
        gt_bbox = torch.clamp(gt_bbox,min=0,max=args.size-1)

        optimizer.zero_grad()
        pred_bbox = model(image, masks, word_id, word_mask)
        
        loss = 0.
        GIoU_loss = GIoU_Loss(pred_bbox*(args.size-1), gt_bbox, args.size-1)
        loss += GIoU_loss

        gt_bbox_ = xyxy2xywh(gt_bbox)
        l1_loss = Reg_Loss(pred_bbox, gt_bbox_/(args.size-1))
        #loss  += l1_loss
        loss.backward()
        optimizer.step()

        wandb.log({
            'loss': loss.item(),
            'l1_loss': l1_loss.item(),
            'GIoU_loss': GIoU_loss.item(),
            'train_iteration': train_iter
            }
        )

        ## box iou
        pred_bbox = torch.cat([pred_bbox[:,:2]-(pred_bbox[:,2:]/2), pred_bbox[:,:2]+(pred_bbox[:,2:]/2)], dim=1)
        pred_bbox = pred_bbox * (args.size-1)
        iou = bbox_iou(pred_bbox.data.cpu(), gt_bbox.data.cpu(), x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size

        ## compute loss
        acc.update(accu, imgs.size(0))   
        miou.update(torch.mean(iou).item(), imgs.size(0))
        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1_loss.item(), imgs.size(0))
        GIoU_losses.update(GIoU_loss.item(), imgs.size(0))
        
        if batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
            'L1_Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})\t' \
            'GIoU_Loss {GIoU_loss.val:.4f} ({GIoU_loss.avg:.4f})\t' \
            'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
            'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
            'vis_lr {vis_lr:.8f}\t' \
            'lang_lr {lang_lr:.8f}\t' \
            .format( \
                epoch, batch_idx, len(train_loader), \
                loss=losses, l1_loss = l1_losses, \
                GIoU_loss = GIoU_losses, miou=miou, acc=acc, \
                vis_lr = optimizer.param_groups[0]['lr'], lang_lr = optimizer.param_groups[2]['lr'])
            print(print_str)
    wandb.log(
        {
            'Epoch_loss': losses.avg,
            'Epoch_l1_loss': l1_losses.avg,
            'Epoch_GIoU': GIoU_losses.avg,
            'IoU': miou.avg,
            'Acc': acc.avg,
            'Epoch': epoch
        }
    )
    return train_iter
    

def validate_epoch(val_loader, model, epoch, wandb_table):
    losses = AverageMeter()
    acc = AverageMeter()
    miou = AverageMeter()

    model.eval()
    print(datetime.datetime.now())
    
    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:,:,:,0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        

        with torch.no_grad():
            pred_bbox = model(image, masks, word_id, word_mask)

        gt_bbox = bbox
        loss = 0.
        GIoU_loss = GIoU_Loss(pred_bbox*(args.size-1), gt_bbox, args.size-1)
        loss += GIoU_loss
        gt_bbox_ = xyxy2xywh(gt_bbox)
        l1_loss = Reg_Loss(pred_bbox, gt_bbox_/(args.size-1))
        #loss  += l1_loss
        losses.update(loss.item(), imgs.size(0))


        pred_bbox = torch.cat([pred_bbox[:,:2]-(pred_bbox[:,2:]/2), pred_bbox[:,:2]+(pred_bbox[:,2:]/2)], dim=1)
        pred_bbox = pred_bbox * (args.size-1)
        

        ## metrics
        iou = bbox_iou(pred_bbox.data.cpu(), gt_bbox.data.cpu(), x1y1x2y2=True)
        # accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/args.batch_size
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size

        acc.update(accu, imgs.size(0))
        # acc_center.update(accu_center, imgs.size(0))
        miou.update(torch.mean(iou).item(), imgs.size(0))


        if batch_idx % args.print_freq == 0:
            print_str = '[' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                .format( \
                    acc=acc, miou=miou,
                    losses = losses)
            print(print_str)
    wandb.log(
        {
            'Epoch_val_loss': losses.avg,
            'Val_IoU': miou.avg,
            'Val_Acc': acc.avg,
            'Epoch': epoch
        }
    )
    for i in range(len(pred_bbox)):
        wandb_table.add_data(epoch+1, np.array(bbox[i].to('cpu')), np.array(pred_bbox[i].to('cpu')))
    return miou.avg
                                                                                                   

if __name__ == "__main__":
    main()