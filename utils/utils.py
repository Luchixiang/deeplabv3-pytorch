import torch
import torch.nn as nn
import numpy as np


def weight_decay(network, l2_value):  # 权重衰退
    decay, no_decay = [], []
    for name, param in network.parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


def image2color(img):
    label_to_color = {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
    }
    img_height, img_width = img.shape()
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for height in range(img_width):
            label = img[row][height]
            img_color[row][height] = label_to_color[label]

    return img_color