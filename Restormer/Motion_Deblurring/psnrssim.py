import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
import argparse
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import utils


def proc(filename):
    try:
        # 处理图像的代码
        tar, prd = filename
        tar_img = utils.load_img(tar)
        prd_img = utils.load_img(prd)

        # PSNR = utils.calculate_psnr(tar_img, prd_img, 0, test_y_channel=True)
        PSNR = utils.calculate_psnr(tar_img, prd_img)
        SSIM = utils.calculate_ssim(tar_img, prd_img)
        return PSNR, SSIM

    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"Error processing {filename}: {e}")
        return None


if __name__ == '__main__':
    gt_path = "C:\\Users\\13086\\Desktop\\webdemo\\image\\Set5\\swinIR"
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')))    # 获取该路径下的目标文件列表 gt_list
    assert len(gt_list) != 0, "Target files not found"

    file_path = "C:\\Users\\13086\\Desktop\\webdemo\\image\\Set5\\original"
    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.tif')))
    assert len(path_list) != 0, "Predicted files not found"

    psnr, ssim = [], []
    img_files = [(i, j) for i, j in zip(gt_list, path_list)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            psnr.append(PSNR_SSIM[0])
            print(psnr)
            ssim.append(PSNR_SSIM[1])

        avg_psnr = sum(psnr) / len(psnr)
        avg_ssim = sum(ssim) / len(ssim)

        print('PSNR: {:f} \n SSIM: {:f}'.format( avg_psnr, avg_ssim))
        # print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))