import os
import numpy as np
from glob import glob
import torch.nn as nn
import torch
import torch.nn.functional as F
from natsort import natsorted
from skimage import io
import cv2
import argparse
from skimage.metrics import structural_similarity
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import concurrent.futures
from Restormer.Motion_Deblurring import utils
from Restormer.basicsr.utils.imresize import imresize
from LAM.ModelZoo import load_model


def proc(filename):
    try:
        # 处理图像的代码
        tar, prd = filename
        tar_img = utils.load_img(tar)  #未处理前的
        prd_img = utils.load_img(prd)  #处理后的

        #PSNR = utils.calculate_psnr(tar_img, prd_img, 0, test_y_channel=True)
        PSNR = utils.calculate_psnr(tar_img, prd_img)
        SSIM = utils.calculate_ssim(tar_img, prd_img)
        return PSNR, SSIM

    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"Error processing {filename}: {e}")
        return None

def deal(choice):
    dataset = "urban100"
    factor = 4
    factor_test = 2
    files = natsorted(glob(os.path.join(dataset, '*.png'))) #返回一个经过自然排序后的png文件名列表
    result = os.path.join("result", choice)
    if not os.path.exists(result):
        os.makedirs(result)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()  # 释放GPU内存
            img = utils.load_img(file_)
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
            input_ =input_.to(device)
            #print(input_.shape)


            model_restoration = load_model(choice+"@Base").to(device)

            restored = model_restoration(input_).to(device)# 输入图像 input_ 送入 model_restoration 模型中进行处理，得到恢复后的图像
            output = torch.nn.functional.interpolate(restored, size=input_.size()[2:], mode='bilinear', align_corners=False)
            #
            # #print(output.shape)
            #
            output = output[:, :, :h_old, :w_old]
            #
            output = torch.clamp(output, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            save_file = os.path.join(result, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(output))
            # restored = restored[:, :, :h_old, :w_old]
            #
            # restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            #
            # save_file = os.path.join(result, os.path.split(file_)[-1])
            # utils.save_img(save_file, img_as_ubyte(restored))



def evaluate(choice):
    file_path = 'urban100'
    path_list = natsorted(glob(os.path.join(file_path, '*.png')))
    deal(choice)
    result_path = os.path.join("result", choice)
    result_list = natsorted(glob(os.path.join(result_path, '*.png')))

    psnr, ssim = [], []
    img_files = [(i, j) for i, j in zip(path_list, result_list)]
    #print(img_files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            psnr.append(PSNR_SSIM[0])
            ssim.append(PSNR_SSIM[1])
        #print(psnr)
    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)

    return avg_psnr, avg_ssim

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     #deal("RCAN")
#     print(evaluate("DBPN"))