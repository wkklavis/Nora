import numpy as np
import torch


def normalize_image(img):
    """
    :param img: b, c, h, w
    """
    img = np.array(img)
    for b in range(img.shape[0]):
        for c in range(img.shape[1]):
            img[b, c] = (img[b, c] - img[b, c].mean()) / img[b, c].std()
    return torch.from_numpy(img)


def normalize_image_to_0_1(img):
    return (img-img.min())/(img.max()-img.min())


def normalize_image_to_m1_1(img):
    return -1 + 2 * (img-img.min())/(img.max()-img.min())
