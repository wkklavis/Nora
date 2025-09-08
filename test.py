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

from importlib import import_module
from segment_anything import sam_model_registry


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

def save_image(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    print("\nSave images")
    print("--" * 15)
    last_name = None
    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
            x, y, path = data['data'], data['mask'], data['name']
            current_name = path
            if last_name is None:
                last_name = path

            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)

            x = x.cuda()
            seg_logit = model(x, False, args.img_size)
            seg_output = torch.sigmoid(seg_logit['masks'].detach().cpu())
            if current_name != last_name:  # Calculate the previous 3D volume
                metrics = calculate_metrics(seg_output3D, y3D)
                del seg_output3D
                del y3D

            try:
                seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_output3D = seg_output.unsqueeze(2)
                y3D = y.unsqueeze(2)

            output_path = os.path.join(snapshot_path, 'saved_imges',str(path[0]).split('/')[-2])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
            draw_x = (x.detach().cpu().numpy() * 255).astype(np.uint8)
            draw_gt = (y.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(output_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_pred.png',
                        draw_output[0][0])
            cv2.imwrite(output_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_x.png',
                        draw_x[0][0])
            cv2.imwrite(output_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_gt.png',
                        draw_gt[0][0])
            last_name = current_name
    logging.info("saved")
    return

def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'PROSTATE': {
            'Dataset': args.dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size, _ = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
