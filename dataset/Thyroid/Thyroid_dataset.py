import torch
from torch.utils.data import Dataset

import os
import json
import numpy as np
from PIL import Image

from torchvision import transforms

from dataset.transform import joint_transforms

from scipy.ndimage.morphology import distance_transform_edt


class TN3KDataset(Dataset):
    def __init__(self, root, list_path, mode, num_class, joint_augment=None, augment=None, target_augment=None, lb_change=True):
        self.joint_augment = joint_augment
        self.augment = augment
        self.lb_change = lb_change
        self.num_class = num_class
        self.target_augment = target_augment
        self.root = root
        if mode == "train":
            self.img_dir = os.path.join(root, 'trainval-image')
            self.lb_dir = os.path.join(root, 'trainval-mask')
            with open(list_path, 'r') as f:
                data = json.load(f)
            self.nlist = [f"{item:04d}" for item in data.get('train', [])]

        else:
            self.img_dir = os.path.join(root, 'test-image')
            self.lb_dir = os.path.join(root, 'test-mask')
            self.nlist = [name.strip().split('.')[0] for name in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.nlist)

    def __getitem__(self, index):
        img_name = str(self.nlist[index])
        # load image
        img_file = os.path.join(self.img_dir, img_name + '.jpg')
        img = Image.open(img_file)
        img = np.array(img).astype(np.uint8)
        if len(img.shape) < 3:
            img = img[:,:,None].repeat(3, axis=2)
        lbl_file = os.path.join(self.lb_dir, img_name + '.jpg')
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl)
        if self.lb_change:
            lbl[lbl > 0] = 1
        if len(lbl.shape) == 2:
            lbl = np.array(lbl[:, :, None]).astype(np.float32)
        else:
            lbl = (np.array(lbl)[:, :, 0][:, :, None]).astype(np.float32)
        # plt.imshow(lbl[...,1])
        # plt.show()
        # print(np.amax(lbl))
        # lbl = (np.array(lbl)[:, :, 0][:, :, None]).astype(np.float32)
        if self.joint_augment is not None:
            img, lbl = self.joint_augment(img, lbl)
        if self.augment is not None:
            img = self.augment(img)
        if self.target_augment is not None:
            lbl = self.target_augment(lbl)
        img = img / 255.0
        gt = lbl[0]
        data = {"image": img,
                "mask": (gt.unsqueeze(0),
                      self.mask_to_edges(gt)),
                "name": img_name}
        return data
    def mask_to_edges(self, mask):
        _edge = mask
        _edge = self.mask_to_onehot(_edge, self.num_class+1)#add 1 to get the gt edge
        _edge = self.onehot_to_binary_edges(_edge, num_classes=self.num_class+1)
        return torch.from_numpy(_edge).float()

    def mask_to_onehot(self, mask, num_classes=3):
        _mask = [mask == i for i in range(1, num_classes)]
        _mask = [np.expand_dims(x, 0) for x in _mask]
        return np.concatenate(_mask, 0)

    def onehot_to_binary_edges(self, mask, radius=2, num_classes=3):
        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

        edgemap = np.zeros(mask.shape[1:])

        for i in range(num_classes - 1):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap

class DDTIDataset(Dataset):
    def __init__(self, root, num_class, joint_augment=None, augment=None, target_augment=None, lb_change=True):
        self.joint_augment = joint_augment
        self.augment = augment
        self.lb_change = lb_change
        self.num_class = num_class
        self.target_augment = target_augment
        self.root = root
        self.img_dir = os.path.join(root, 'p_image')
        self.lb_dir = os.path.join(root, 'p_mask')

        self.nlist = [name.strip().split('.')[0] for name in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.nlist)

    def __getitem__(self, index):
        img_name = str(self.nlist[index])
        # load image
        img_file = os.path.join(self.img_dir, img_name + '.PNG')
        img = Image.open(img_file)
        img = np.array(img).astype(np.uint8)
        if len(img.shape) < 3:
            img = img[:,:,None].repeat(3, axis=2)
        lbl_file = os.path.join(self.lb_dir, img_name + '.PNG')
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl)
        if self.lb_change:
            lbl[lbl > 0] = 1
        if len(lbl.shape) == 2:
            lbl = np.array(lbl[:, :, None]).astype(np.float32)
        else:
            lbl = (np.array(lbl)[:, :, 0][:, :, None]).astype(np.float32)
        # plt.imshow(lbl[...,1])
        # plt.show()
        # print(np.amax(lbl))
        # lbl = (np.array(lbl)[:, :, 0][:, :, None]).astype(np.float32)
        if self.joint_augment is not None:
            img, lbl = self.joint_augment(img, lbl)
        if self.augment is not None:
            img = self.augment(img)
        if self.target_augment is not None:
            lbl = self.target_augment(lbl)
        img = img / 255.0
        gt = lbl[0]
        data = {"image": img,
                "mask": (gt.unsqueeze(0),
                      self.mask_to_edges(gt)),
                "name": img_name}
        return data
    def mask_to_edges(self, mask):
        _edge = mask
        _edge = self.mask_to_onehot(_edge, self.num_class+1)
        _edge = self.onehot_to_binary_edges(_edge, num_classes=self.num_class+1)
        return torch.from_numpy(_edge).float()

    def mask_to_onehot(self, mask, num_classes=3):
        _mask = [mask == i for i in range(1, num_classes)]
        _mask = [np.expand_dims(x, 0) for x in _mask]
        return np.concatenate(_mask, 0)

    def onehot_to_binary_edges(self, mask, radius=2, num_classes=3):
        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

        edgemap = np.zeros(mask.shape[1:])

        for i in range(num_classes - 1):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap

if __name__ == '__main__':
    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    path = '/home/data/Dataset_BUSI_with_GT/BUSI_WO_normal'
    train_json = '/home/data/Dataset_BUSI_with_GT/BUSI_WO_normal/train_list.json'
    with open(train_json, 'r') as f:
        nlist = json.load(f)
    train_joint_transform = joint_transforms.Compose([
        transforms.ToPILImage(),
        joint_transforms.FreeScale((256, 256))
    ])
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    target_transform = transforms.Compose([
        transforms.ToTensor()])
    train_set = TN3KDataset(root='cfg.DATASET.DATA_DIR', img_list='cfg.DATASET.TRAIN_LIST',
                            joint_augment=train_joint_transform,
                            augment=transform, target_augment=target_transform)
    data = torch.utils.data.DataLoader(train_set, batch_size=1)
    for i, data_ in enumerate(data):
        img, lb = data_
        # plt.imshow(img)
        for j in range(3):
            means[j] += np.array(img[0, j, :, :]).mean()
            stdevs[j] += np.std(np.array(img[0, j, :, :]), ddof=1)
        print(i+1)
    means = np.asarray(means) / (i+1)
    stdevs = np.asarray(stdevs) / (i+1)

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
    print("=============")
    print(np.mean(means))
    print(np.mean(stdevs))
