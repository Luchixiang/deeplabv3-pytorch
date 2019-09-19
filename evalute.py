# import torch
# import torch.nn as nn
# # from utils.utils import image2color
# import numpy as np
# from model.deeplabv3 import DeepLabv3
# import os
# import cv2
#
# batch_size = 1
# network = DeepLabv3('eval', os.getcwd()).cuda()
# #network.load_state_dict(torch.load('./pretrained_mode/deeplabv3_pretrained.pth'))
#
# loss_fn = nn.CrossEntropyLoss()
# network.eval()
#
# with torch.no_grad():
#     imgs = torch.from_numpy(imgs).cuda()
#     imgs = imgs.float()
#     # label_img =
#     output = network(imgs)
#     # loss = loss_fn()
#     output = output.data.cpu().numpy()
#     pred_label_imgs = np.argmax(output, axis=1)  # (shape: (batch_size, img_h, img_w))
#     pred_label_imgs = pred_label_imgs.astype(np.uint8)
