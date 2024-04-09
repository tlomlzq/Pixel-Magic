import os
import torch
import importlib.util
import yaml
import yaml as yaml_module
import streamlit as st
from upload import ModelData
#from demo import initialize_session_state


MODEL_DIR = 'LAM/ModelZoo/models'


MODEL_KWARGS_DICT = {
  "RCAN" : {"factor":4, "num_channels":3},
  "CARN" : {"factor":4, "num_channels":3},
  "RRDBNET":{"num_in_ch":3, "num_out_ch":3},
  "SAN": {"factor":4, "num_channels":3},
  "RNAN":{"factor":4, "num_channels":3},
  "EDSR":{"factor":4, "num_channels":3},
  "HAT":{"upscale":4, "in_chans":3, "img_size":64,
        "embed_dim":180, "window_size":16,
        "compress_ratio":3, "squeeze_factor": 30,
        "conv_scale": 0.01, "overlap_ratio": 0.5, "img_range": 1.,
        "depths":[6, 6, 6, 6, 6, 6],
        "num_heads":[6, 6, 6, 6, 6, 6],
        "mlp_ratio":2,
        "upsampler": 'pixelshuffle',
        "resi_connection": '1conv'
         },

  "DBPN":{"num_channels":3, "base_filter":64, "feat":256,
          "num_stages":7, "scale_factor":3},
  'SWINIR':{
      "upscale":4,
      'in_chans':3,
      "img_size":48,
      "window_size":8,
      "img_range":1.,
      "depths":[6, 6, 6, 6, 6, 6],
      "embed_dim":180,
      "num_heads":[6, 6, 6, 6, 6, 6],
      "mlp_ratio":2,
      "upsampler": 'pixelshuffle',
      "resi_connection": '1conv'
  }
}


def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))

def parse_yaml(yaml_path: str):
    with open(yaml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def get_network(model_name: str, Model_list):
    from importlib import import_module
    model_list = Model_list
    model_base_name = model_name.split('-')[0]
    print(f'Getting SR Network {model_name}')

    yaml_folder = "LAM/ModelZoo/yaml"  # yaml文件夹路径

    if model_base_name in model_list:
        try:
            # 导入需要的模块和类
            module_name = "LAM.ModelZoo.NN." + model_base_name.lower()
            net_module = import_module(module_name)
            if model_base_name =='SWINIR':
                class_name = 'SwinIR'
            else:
                class_name = model_base_name
            #print(class_name)
            net_class = getattr(net_module, class_name)

            if model_base_name.upper() in MODEL_KWARGS_DICT:
                # 从字典中获取模型参数
                kwargs = MODEL_KWARGS_DICT.get(model_base_name.upper(), {})
                net = net_class(**kwargs)
            else:
                # 如果在字典中找不到，就去yaml文件中查找
                yaml_path = os.path.join(yaml_folder, f"{model_base_name.lower()}.yml")
                # print(yaml_path)
                with open(yaml_path, mode='r', encoding='utf-8') as f:
                    x = yaml_module.load(f, Loader=yaml_module.FullLoader)
                    # network_g_type = x['network_g'].pop('type')
                    # print("Network G type:", network_g_type)
                    s = x['network_g'].pop('type')
                    model_class = getattr(net_module, s)
                    # 使用上述参数创建模型实例
                    net = model_class(**x['network_g'])
                # config = parse_yaml(yaml_path)
                #
                # kwargs = config.get(model_base_name, {})

            return net
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    else:
        print(f"Model {model_name} is not supported")
        return None

def load_model(model_loading_name, Model_list, Model_pth_list):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    MODEL_LIST = ['RCAN', 'CARN', 'RRDBNet', 'SAN', 'EDSR', 'HAT', 'SWINIR']
    model_list = Model_list
    model_pth_list = Model_pth_list
    model_path = MODEL_DIR
    # model_path = "models"
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()

    assert model_name in model_list or model_name in model_pth_list.keys(), 'check your model name before @'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = get_network(model_name, model_list).to(device)
    state_dict_path = os.path.join(model_path, model_pth_list[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    #state_dict = torch.load(state_dict_path, map_location='cpu')
    if model_name == 'SWINIR':
        param_key_g = 'params'
        pretrained_model = torch.load(state_dict_path)
        net.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    elif model_name == 'HAT':
        param_key_g = 'params_ema'
        pretrained_model = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)
    elif model_name not in MODEL_LIST:
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict['params'])
    else:
        state_dict = torch.load(state_dict_path, map_location='cpu')  # 加载权重
        net.load_state_dict(state_dict)
    return net
