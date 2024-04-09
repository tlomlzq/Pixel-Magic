import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
#from restormerarch import Restormer
import argparse
from pdb import set_trace as stx
import numpy as np

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(task, parameters):
    global weights
    if task == '运动去模糊':
        weights = 'Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth'
        #weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == '单图像散焦去模糊':
        weights = 'Restormer/Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth'
        #weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == '图像去雨':
        weights = 'Restormer/Motion_Deblurring/pretrained_models/deraining.pth'
        #weights = 'Motion_Deblurring/pretrained_models/deraining.pth'
        # weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'deraining.pth')
    # elif task == 'Real_Denoising':
    #     weights = 'Restormer/Motion_Deblurring/pretrained_models/real_denoising.pth'
    #     #weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'real_denoising.pth')
    #     parameters['LayerNorm_type'] =  'BiasFree'
    elif task == '高斯彩色去噪':
        weights = 'Restormer/Motion_Deblurring/pretrained_models/gaussian_color_denoising_blind.pth'
        #weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == '高斯灰度去噪':
        weights = 'Restormer/Motion_Deblurring/pretrained_models/gaussian_gray_denoising_blind.pth'
        #weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] =  1
        parameters['out_channels'] =  1
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters

def main(inp_path, choice):
    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
    inp_dir = inp_path
    # out_dir = out_path
    if any([inp_dir.endswith(ext) for ext in extensions]):
        files = [inp_dir]

    else:
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(inp_dir, '*.' + ext)))
        files = natsorted(files)

    if len(files) == 0:
        raise Exception(f'No files found at {inp_dir}')

    if choice == 'Deraining':
        weight = 'Restormer/Motion_Deblurring/pretrained_models/deraining.pth'

        ####### Load yaml #######
        yaml_file = 'Restormer/Deraining/Options/Deraining_Restormer.yml'
        import yaml

        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
        s = x['network_g'].pop('type')

        load_arch = run_path('Restormer/basicsr/models/archs/restormer_arch.py')
        model_restoration = load_arch['Restormer'](**x['network_g'])

        checkpoint = torch.load(weight)
        model_restoration.load_state_dict(checkpoint['params'])
        model_restoration.cuda()
        model_restoration = nn.DataParallel(model_restoration)
        model_restoration.eval()

        factor = 8

        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                img = np.float32(utils.load_img(inp_dir)) / 255.
                img = torch.from_numpy(img).permute(2, 0, 1)
                input_ = img.unsqueeze(0).cuda()

                # Padding in case images are not multiples of 8
                h, w = input_.shape[2], input_.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                restored = model_restoration(input_)

                # Unpad images to original dimensions
                restored = restored[:, :, :h, :w]

                restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    else:
        # Get model weights and parameters
        parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                      'num_refinement_blocks': 4, 'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False,
                      'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
        weight, parameters = get_weights_and_parameters(choice, parameters)

        # load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
        load_arch = run_path('Restormer/basicsr/models/archs/restormer_arch.py')
        #load_arch = run_path('basicsr/models/archs/restormer_arch.py')
        model = load_arch['Restormer'](**parameters)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint['params'])
        model.eval()

        img_multiple_of = 8

        with torch.no_grad():
            for file_ in tqdm(files):
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()

                if choice == '高斯灰度去噪':
                    img = load_gray_img(file_)
                else:
                    img = load_img(file_)

                input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)

                # Pad the input if not_multiple_of 8
                height, width = input_.shape[2], input_.shape[3]
                H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
                padh = H - height if height % img_multiple_of != 0 else 0
                padw = W - width if width % img_multiple_of != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                restored = model(input_)
                restored = torch.clamp(restored, 0, 1)

                # Unpad the output
                restored = restored[:, :, :height, :width]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = img_as_ubyte(restored[0])


    out_dir = "Restormer/demo/restored"

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    # stx()
    if choice == '高斯灰度去噪':
        saved_image_path = os.path.join(out_dir, f + '.png')
        save_gray_img(saved_image_path, restored)
        # save_gray_img((os.path.join(out_dir, f + '.png')), restored)

    elif choice =='图像去雨':
        saved_image_path = os.path.join(out_dir, os.path.splitext(os.path.split(inp_dir)[-1])[0] + '.png')
        utils.save_img(saved_image_path, img_as_ubyte(restored))
    else:
        saved_image_path = os.path.join(out_dir, f + '.png')
        save_img(saved_image_path, restored)
        # save_img((os.path.join(out_dir, f + '.png')), restored)

    return saved_image_path


