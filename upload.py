import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
import streamlit as st
import concurrent.futures

from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx
from LAM.size import imresize
from Restormer.basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import cv2
import yaml as yaml_module
import utils
from LPIPS.lp import lp

class ModelData:
    def __init__(self):
        self.MODEL_DIR = 'LAM/ModelZoo/models'
        self.NN_LIST = ['RCAN', 'CARN', 'RRDBNet', 'SAN', 'EDSR', 'HAT', 'SWINIR']
        self.MODEL_LIST = {
            'RCAN': {'Base': 'RCAN.pt'},
            'CARN': {'Base': 'CARN_7400.pth'},
            'RRDBNet': {'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth'},
            'SAN': {'Base': 'SAN_BI4X.pt'},
            'EDSR': {'Base': 'EDSR-64-16_15000.pth'},
            'HAT': {'Base': 'HAT_SRx4_ImageNet-pretrain.pth'},
            'SWINIR': {'Base': "SwinIR.pth"}

        }
        self.metrics = {
            'RCAN': {'psnr': 32.63, 'ssim': 0.9002, 'lpips': 0.1692},
            'CARN': {'psnr': 32.13, 'ssim': 0.8937, 'lpips': 0.1792},
            'RRDBNet': {'psnr': 32.60, 'ssim': 0.9002, 'lpips': 0.1698},
            'SAN': {'psnr': 32.64, 'ssim': 0.9003, 'lpips': 0.1689},
            'EDSR': {'psnr': 32.46, 'ssim': 0.8968, 'lpips': 0.1725},
            'HAT': {'psnr': 33.18, 'ssim': 0.9037, 'lpips': 0.1618},
            'SWINIR': {'psnr': 32.72, 'ssim': 0.9021, 'lpips': 0.1681}
        }

    def proc(self, filename):
        try:
            # 处理图像的代码
            tar, prd = filename
            tar_img = utils.load_img(tar)
            prd_img = utils.load_img(prd)

            PSNR = utils.calculate_psnr(tar_img, prd_img, 0, test_y_channel=True)
            #PSNR = utils.calculate_psnr(tar_img, prd_img)
            SSIM = utils.calculate_ssim(tar_img, prd_img, 0, test_y_channel=True)
            return PSNR, SSIM

        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"Error processing {filename}: {e}")
            return None


    def update(self, weight_path, yaml_path, arch_path):
        weight = weight_path
        yaml_file = yaml_path
        arch_file = arch_path

        with open(yaml_file, mode='r', encoding='utf-8') as f:
            x = yaml_module.load(f, Loader=yaml_module.FullLoader)

            s = x['network_g'].pop('type')

            self.NN_LIST.append(s)
            # print(NN_LIST)
            model_list = self.NN_LIST
            file_name = os.path.basename(weight)
            self.MODEL_LIST[s] = {
                'Base': file_name
            }
            model_pth_update = self.MODEL_LIST

            import importlib.util
            arch_name = os.path.basename(arch_file)
            spec = importlib.util.spec_from_file_location(arch_name, arch_file)
            arch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arch)

            model_class = getattr(arch, s)
            model = model_class(**x['network_g'])
            checkpoint = torch.load(weight)
            model.load_state_dict(checkpoint['params'])
            model.cuda()
            model = nn.DataParallel(model)
            model.eval()

            # 计算psnr, ssim, lpip并加进去
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            factor = 8
            factor_test = 4
            file_path = 'image/Set5/original'
            result = 'image/Set5'
            result_path = os.path.join(result, s)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            files = natsorted(glob(os.path.join(file_path, '*.png')))
            with torch.no_grad():
                for file_ in tqdm(files):  # 输入图像的预处理
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()  # 释放GPU内存
                    img = utils.load_img(file_)  # 转为RGB
                    img = img.astype(np.float32) / 255.
                    h_old, w_old, _ = img.shape
                    np.random.seed(seed=0)  # for reproducibility
                    img = imresize(img, scalar_scale=1 / factor_test)  # 图像进行缩放以生成低质量（LQ）图像，然后再将其还原回原始尺寸，用于模拟低分辨率输入
                    img = imresize(img, scalar_scale=factor_test)

                    img = torch.from_numpy(img).permute(2, 0,
                                                        1)  # 图像转换为PyTorch张量，将原始张量的第0维（通道维）、第1维（行维）和第2维（列维）重新排列为新的顺序
                    input_ = img.unsqueeze(0).cuda()  # 扩展维度0意味着在最前面添加一个新的维度，从而将其变成形状为 [1, 通道, 行, 列] 的张量,并移动到GPU上

                    h_pad = (h_old // factor + 1) * factor - h_old
                    w_pad = (w_old // factor + 1) * factor - w_old
                    input_ = torch.cat([input_, torch.flip(input_, [2])], 2)[:, :, :h_old + h_pad, :]
                    input_ = torch.cat([input_, torch.flip(input_, [3])], 3)[:, :, :, :w_old + w_pad]
                    input_ = input_.type(torch.cuda.FloatTensor)

                    model = model.to(device)

                    restored = model(input_)

                    # 恢复后图像保存
                    restored = restored[:, :, :h_old, :w_old]
                    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    save_file = os.path.join(result_path, os.path.split(file_)[-1])
                    utils.save_img(save_file, img_as_ubyte(restored))

                # 计算PSNR, SSIM, LPIPS
                file_path = 'image/Set5/original'
                path_list = natsorted(glob(os.path.join(file_path, '*.png')))
                result_list = natsorted(glob(os.path.join(result_path, '*.png')))

                psnr, ssim = [], []
                img_files = [(i, j) for i, j in zip(result_list, path_list)]
                #print(img_files)
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    for filename, PSNR_SSIM in zip(img_files, executor.map(self.proc, img_files)):
                        psnr.append(PSNR_SSIM[0])
                        ssim.append(PSNR_SSIM[1])

                average_psnr = round(sum(psnr) / len(psnr), 2)
                average_ssim = round(sum(ssim) / len(ssim), 4)
                average_lpips = round(lp(file_path, result_path), 4)

                self.metrics[s] = {
                    'psnr': average_psnr,
                    'ssim': average_ssim,
                    'lpips': average_lpips
                }
                metrics_update = self.metrics

        return model_list, model_pth_update, metrics_update

# if __name__ == "__main__":
#     weight = 'new_model/weight/restormer.pth'
#     yaml_file_path = 'new_model/yaml/restormer.yml'
#     arch = 'new_model/arch/restormer.py'
#     my_object = ModelData()
#     my_object.update(weight, yaml_file_path, arch)
#     print(my_object.metrics)