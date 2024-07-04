# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

# import os
import pandas as pd 
from pathlib import Path
import torch
import numpy as np
import util.misc as misc
from torchvision.utils import save_image
from torchvision import transforms, datasets
from PIL import Image
import cv2
import openpyxl
from sklearn.manifold import TSNE

def train_one_epoch(model, train_loaders, optimizer, optimizer_domain, epoch, args, total_len, logger):
    
    model.train()
    model.frozen_cls()
    
    for bid, batch_data in enumerate(train_loaders):
        if bid == total_len:
            break

        if args.sample_pos:
            images = torch.cat([x for x, y, x_pos in batch_data])
            labels = torch.cat([y for x, y, x_pos in batch_data])
            images_pos = torch.cat([x_pos for x, y, x_pos in batch_data]) # positive sample
        else:
            images = torch.cat([x for (ori_x, x), y in batch_data])
            labels = torch.cat([y for _, y in batch_data])
            ori_images = torch.cat([ori_x for (ori_x, x), y in batch_data])

        images = images.cuda()
        labels = labels.cuda()
        status, _, _  = model(images, labels, args.mask_ratio)
        
        total_loss = status['total_loss']
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if bid%(total_len//2)==0 and bid!=0:
            logger.info("Bid:[{}/{}], Recon_Loss:{:.4f}, Ada_Loss:{:.4f}, Pred_loss:{:.4f}, Total_Loss:{:.4f}".format(
                        bid, total_len, status['recon_loss'].item(), status['ada_loss'].item(), \
                        status['pred_loss'].item(),status['total_loss'].item()))
            
    domain_labels = torch.zeros([args.batch_size*3], dtype=torch.int64).cuda()
    domain_labels[args.batch_size:args.batch_size*2] = 1
    domain_labels[args.batch_size*2:args.batch_size*3] = 2
    criterion = torch.nn.CrossEntropyLoss()
    
    num = 0; acc_s_num = 0
    if (epoch%args.inter_ada==0 or epoch<=5) and epoch<=args.max_epoch_ada:
        model.frozen_backbone()
        for bid, batch_data in enumerate(train_loaders):
            if bid == total_len:
                break
 
            if args.sample_pos:
                images = torch.cat([x for x, y, x_pos in batch_data])
                labels = torch.cat([y for x, y, x_pos in batch_data])
                images_pos = torch.cat([x_pos for x, y, x_pos in batch_data]) # positive sample
            else:
                images = torch.cat([x for (ori_x, x), y in batch_data])
                labels = torch.cat([y for _, y in batch_data])
                ori_images = torch.cat([ori_x for (ori_x, x), y in batch_data])
            
            images = images.cuda()
            labels = labels.cuda()
            pred_domain = model.pred_domain(images)

            pred_domain_loss = criterion(pred_domain, domain_labels)
            loss = pred_domain_loss

            optimizer_domain.zero_grad()
            loss.backward()
            optimizer_domain.step()

            num += domain_labels.size(0)
            acc_s_num += (pred_domain.argmax(dim=1) == domain_labels).sum().item()

            if bid%(total_len//2)==0 and bid!=0:
                logger.info("Pred_domain_loss: {:.5f}, Acc:{:.2f}%".format(pred_domain_loss.item(), acc_s_num/num*100))
    
