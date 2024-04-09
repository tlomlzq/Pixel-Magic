import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append('LAM_Demo-main')
from LAM.ModelZoo.utils import Tensor2PIL, PIL2Tensor
from LAM.ModelZoo import load_model
from LAM.SaliencyModel.utils import vis_saliency, vis_saliency_kde, grad_abs_norm, prepare_images, make_pil_grid
from LAM.SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from LAM.SaliencyModel.attributes import attr_grad
from LAM.SaliencyModel.BackProp import attribution_objective, Path_gradient
from LAM.SaliencyModel.BackProp import saliency_map_PG as saliency_map
from LAM.SaliencyModel.BackProp import GaussianBlurPath
import random

def main(inp_path, choice):
    model = load_model(choice+"@Base")

    window_size = 16  # Define windoes_size of D
    img_lr, img_hr = prepare_images(inp_path)  # Change this image name
    tensor_lr = PIL2Tensor(img_lr)[:3]
    tensor_hr = PIL2Tensor(img_hr)[:3]
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2)
    cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

    plt.imshow(cv2_hr)

    w, h = 110, 150

    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)

    sigma = 1.2
    fold = 50
    l = 9
    alpha = 0.5
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)

    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective,
                                                                              gaus_blur_path_func, cuda=True)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
         saliency_image_abs,
         blend_abs_and_input,
         blend_kde_and_input,
         Tensor2PIL(torch.clamp(torch.Tensor(result), min=0., max=1.))]
    )
    recovered_hr_image = Tensor2PIL(torch.clamp(torch.Tensor(result_numpy), min=0., max=1.))

    torch.cuda.empty_cache()

    file_name = os.path.basename(inp_path)
    save_folder = "C:\\Users\\13086\\Desktop\\webdemo\\LAM\\image\\RNAN"
    save_path = os.path.join(save_folder, file_name)

    recovered_hr_image.save(save_path, "PNG", compress_level=0)


def process_images_in_folder(folder_path, choice):
    image_files = os.listdir(folder_path)
    for image_file in image_files:
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # 可以根据实际情况调整支持的图像格式
            image_path = os.path.join(folder_path, image_file)
            main(image_path, choice)  # 调用处理单张图像的 main 函数



if __name__ == "__main__":
    folder_path = "C:\\Users\\13086\\Desktop\\webdemo\\LAM\\urban100"
    process_images_in_folder(folder_path, "CARN")