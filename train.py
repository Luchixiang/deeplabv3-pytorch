import torch
import numpy as np
import torch.nn as nn
import os
from model.deeplabv3 import DeepLabv3
import torch.nn.functional as F
import cv2
import time
import matplotlib.pyplot as plt
from utils.utils import weight_decay

model_id = '1'
num_epochs = 1000
batch_size = 3
learning_rate = 0.0001

network = DeepLabv3(model_id=model_id, project_dir=os.getcwd()).cuda()
params = weight_decay(network, 0.0001)
optimizer = torch.optim.Adam(params, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    print('start new epoch')
    print("epoch: %d/%d" % (epoch + 1, num_epochs))
    network.train()
    batch_loss = []


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power): # ploy_learning_rate in paper
    if iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
