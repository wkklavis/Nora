import os
import sys

import cv2
from tqdm import tqdm
import logging
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset.transform.normalize import normalize_image
from segment_anything.utils.metrics import calculate_metrics, dice, jaccard, avg_surface_distance, hausdorff_distance


def inference(args, epoch, snapshot_path, test_loader, model, dataset_name=''):
    print("\nTesting and Saving the results...")
    print("--" * 15)
    iou_all = 0
    dice_all = 0
    asd_all = 0
    hd_all = 0
    num = 0
    ignore = 0
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
            img, lb = data["image"], data["mask"][0]
            img = normalize_image(img)
            img, lb = img.cuda(), lb.cuda()

            lb[lb > 0] = 1
            num += 1
            outputs = model(img, False, args.img_size)
            output_masks = outputs['masks']
            net_out = output_masks>0
            pred = net_out.float().int()
            pred = np.array(pred.cpu(), dtype=np.uint8)
            lb = np.array(lb.cpu(), dtype=np.uint8)

            iou_all += jaccard(pred, lb)
            dice_all += dice(pred, lb)
            asd_all += avg_surface_distance(pred, lb, nan_for_nonexisting=False)
            hd_all += hausdorff_distance(pred, lb, nan_for_nonexisting=False)

            if pred.sum() == 0 or lb.sum() == 0:
                ignore += 1

    # Calculate the metrics
    Dice = dice_all / num
    mIoU = iou_all / num
    ASD = asd_all / (num-ignore)
    HD = hd_all / (num-ignore)
    with open(snapshot_path + '/' + 'test_' + args.Source_Dataset + '_to'+ '.txt', 'a', encoding='utf-8') as f:
        f.write('Epoch '+str(epoch)+' Test Metrics:\n')
        if dataset_name == 'BUSI' or dataset_name == 'TN3K':
            f.write('IntraDomain '+ dataset_name+' Dice:'+ str(Dice) + ', ' + 'mIoU:' + str(mIoU) + ', ' + 'ASD:' + str(ASD) + ', ' + 'HD:' + str(HD) + '\n\n')  # Dice
        else:
            f.write('CrossDomain ' + dataset_name+' Dice:'+ str(Dice) + ', ' + 'mIoU:' + str(mIoU) + ', ' + 'ASD:' + str(ASD) + ', ' + 'HD:' + str(HD) + '\n')

    if dataset_name == 'BUSI' or dataset_name == 'TN3K':
        logging.info('IntraDomain '+ dataset_name+' Dice:'+ str(Dice) + ', ' + 'mIoU:' + str(mIoU) + ', ' + 'ASD:' + str(ASD) + ', ' + 'HD:' + str(HD)+'\n')
    else:
        logging.info('CrossDomain ' + dataset_name+' Dice:'+ str(Dice) + ', ' + 'mIoU:' + str(mIoU) + ', ' + 'ASD:' + str(ASD) + ', ' + 'HD:' + str(HD) + '\n')
    return Dice, ASD


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict