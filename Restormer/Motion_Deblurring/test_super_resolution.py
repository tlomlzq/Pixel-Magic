## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from arch import Restormer    # 传入模型文件restormer_arch.py
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx
from size import imresize
from Restormer.basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import cv2

input_dir = "Datasets/original"
result_dir = "Datasets/set5"

weight = "pretrained_models/net_g_300000.pth"

yaml_file = 'Options/Classical_SR_Restormer.yml'       # 配置文件路径
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r', encoding='utf-8'), Loader=Loader)       # 传入超参数配置文件.yml

s = x['network_g'].pop('type')
##########################


factor = 8
factor_test = 4

model_restoration = Restormer(**x['network_g'])
checkpoint = torch.load(weight)
model_restoration.load_state_dict(checkpoint['params'])
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

files = natsorted(glob(os.path.join(input_dir, '*.png')))
with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()    # 释放GPU内存
        img = utils.load_img(file_)   #转为RGB
        img = img.astype(np.float32) / 255.
        h_old, w_old, _ = img.shape

        np.random.seed(seed=0)  # for reproducibility

        img = imresize(img, scalar_scale=1 / factor_test)  # 图像进行缩放以生成低质量（LQ）图像，然后再将其还原回原始尺寸，用于模拟低分辨率输入
        img = imresize(img, scalar_scale=factor_test)

        img = torch.from_numpy(img).permute(2, 0, 1)  # 图像转换为PyTorch张量，将原始张量的第0维（通道维）、第1维（行维）和第2维（列维）重新排列为新的顺序
        input_ = img.unsqueeze(0).cuda()  # 扩展维度0意味着在最前面添加一个新的维度，从而将其变成形状为 [1, 通道, 行, 列] 的张量,并移动到GPU上

        h_pad = (h_old // factor + 1) * factor - h_old
        w_pad = (w_old // factor + 1) * factor - w_old
        input_ = torch.cat([input_, torch.flip(input_, [2])], 2)[:, :, :h_old + h_pad, :]
        input_ = torch.cat([input_, torch.flip(input_, [3])], 3)[:, :, :, :w_old + w_pad]
        input_ = input_.type(torch.cuda.FloatTensor)

        #说明输入的图像颜色没有问题

        restored = model_restoration(input_)
        restored = restored[:, :, :h_old, :w_old]

        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        save_file = os.path.join(result_dir, os.path.split(file_)[-1])
        utils.save_img(save_file, img_as_ubyte(restored))