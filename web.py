import streamlit as st
from streamlit_option_menu import option_menu
import json
from streamlit_lottie import st_lottie
from Restormer import restore
from io import StringIO, BytesIO
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
import shutil
from LAM import Lam
from LPIPS import lpips
import yaml
import matplotlib.pyplot as plt
import cv2
from upload import ModelData

def initialize_session_state():
    class SessionState:
        def __init__(self):
            self.last_uploaded_model = None
            self.selected = None  # 在这里添加 selected 属性
            self.model_names = [
                'RCAN',
                'CARN',
                'RRDBNet',
                'SAN',
                'EDSR',
                'HAT',
                'SWINIR'
            ]
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

    return SessionState()

#Layout
st.set_page_config(
    page_title="PixelMagic",
    layout="wide",
    initial_sidebar_state="expanded")

#Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

def click_restore(path, choice):
    img_path = path
    option = choice
    return restore.main(img_path, option)

def click_lam(path, choice, Model_list, Model_pth_list):
    img_path = path
    option = choice
    model_list = Model_list
    model_pth_list = Model_pth_list
    return Lam.main(img_path, option, model_list, model_pth_list)

def click_lpips(dir0,dir1):
    d0 = dir0
    d1 = dir1
    return lpips.main(d0, d1)

def convert_path(path):
    # 将斜杠转换为反斜杠
    converted_path = path.replace("\\", "/")
    return converted_path

def save_image_to_absolute_path(relative_image_path, absolute_output_path):
    current_directory = os.getcwd()
    input_image_path = os.path.join(current_directory, relative_image_path)
    shutil.copyfile(input_image_path, absolute_output_path)

# 获得图片 bytes => [[list]]
def get_upload_img(upload_file):
    bytes_stream = BytesIO(upload_file.getvalue())
    capture_img = Image.open(bytes_stream)
    return cv2.cvtColor(np.asarray(capture_img), cv2.COLOR_RGB2BGR)


# 图片储存
def save_img(img_list):
    now = str(time.time()).split(".")[1]
    name = f'Restormer\\demo\\degraded\\image_{now}.jpg'
    cv2.imwrite(filename=name, img=img_list)
    return name

def save_image(image_path, image):
    img = Image.open(image)
    img.save(image_path)

# 资源删除
def remove_file(path):
    if os.path.exists(path):
        # remove
        os.remove(path)

#Options Menu
with st.sidebar:
    selected = option_menu('PixelMagic', ["Introduce", 'Image Restoration','Upload', 'Perf Comp'],
        icons=['play-btn','image','upload', 'info-circle'],menu_icon='intersect', default_index=0)
    lottie = load_lottiefile("Cartoon/cat.json")
    st_lottie(lottie,key='loc')

def process_image(upload_file):

    if upload_file is not None:
        #st.image(upload_file)
        #st.markdown('Please select a function you want to implement:')
        menu_options = ['Motion_Deblurring', 'Deraining', 'Single_Image_Defocus_Deblurring',
                        'Gray_Denoising', 'Color_Denoising']
        selected_option = st.selectbox('Please select a function you want to implement', menu_options)  # 这个是task
        if st.button('Start recovery'):
            res = get_upload_img(upload_file)
            name = save_img(res)  # 这个是input_dir
            # 处理
            st.write('Recovering. . .')
            restore_res = click_restore(name, selected_option)

            # 显示检测完成消息
            st.write('{} task completed!\n The result is as follows:'.format(selected_option))
            st.image(restore_res)
            st.write('The comparison picture before and after image restoration is as follows:')
            col1, col2 = st.columns(2)
            with col1:
                st.image(name, width=300)

            with col2:
                st.image(restore_res, width=300)
            re_img = convert_path(restore_res)
            # st.write(re_img)
            return re_img

def save(res_img_path, upload_file, save_folder_path):
    # 加载恢复后的图片
    try:
        res_img = Image.open(res_img_path)
    except Exception as e:
        st.error(f"Error loading restored image: {e}")
        return

    # 生成保存图像的文件名
    file_name = upload_file.name.split('.')[0] + '_restored.png'
    save_path = os.path.join(save_folder_path, file_name)

    # 将图像数据保存到文件
    try:
        res_img.save(save_path)
        st.success(f"The recovered image has been successfully saved at: {save_path} 🌟")
    except Exception as e:
        st.error(f"Error saving image: {e}")


def validate_folder_path(folder_path):
    # 检查路径是否为空
    if not folder_path:
        return False, "Folder path cannot be empty. ⚠️"

    # 检查路径是否存在
    if not os.path.exists(folder_path):
        return False, f"Folder path '{folder_path}' does not exist."

    # 检查路径是否是文件夹
    if not os.path.isdir(folder_path):
        return False, f"'{folder_path}' is not a valid folder path."

    return True, ""


def two_page():
    col1, col2 = st.columns([7, 3])
    # 在左侧列添加内容
    with col1:
        st.title("Image Restoration")
        st.write("On this page, you can experience the magic of image restoration. 🤗")
        #st.write("Please upload the pictures that need to be restored")
    # 在右侧列添加内容
    with col2:
        lottie6 = load_lottiefile("Cartoon/sheep.json")
        st_lottie(lottie6, key='come', height=150, width=200)
    #st.title("Image Restoration")
    #st.markdown('<span style="font-size: 20px;">Please upload the pictures that need to be restored</span>',
                #unsafe_allow_html=True)
    #st.write("Please upload the pictures that need to be restored")

    upload_file = st.file_uploader(label='Please upload the pictures that need to be restored', type=['jpg', 'png', 'jpeg'], key="uploader5")

    if upload_file is not None:
        st.image(upload_file)
    save_folder_path = st.text_input("Enter the folder path to save the image", "")

    # 验证文件夹路径
    is_valid_folder_path, folder_path_error = validate_folder_path(save_folder_path)
    if not is_valid_folder_path:
        st.warning(folder_path_error)
        return

    res_img = process_image(upload_file)
    if res_img is not None:
        save(res_img, upload_file, save_folder_path)

def update_session_state():
    session_state = initialize_session_state()

    col1, col2 = st.columns([7, 3])
    # 在左侧列添加内容
    with col1:
        st.title('Upload your model')
        st.markdown(" ")
        st.markdown(" ")

        st.write(
            "If you want to evaluate your own model, you need to upload the weight file, yaml file, and structure file of the model. 🎈")
        st.write("So follow the steps below, brother! 👇")
        st.markdown(" ")
        st.markdown(" ")
    # 在右侧列添加内容
    with col2:
        lottie8 = load_lottiefile("Cartoon/animal.json")
        st_lottie(lottie8, key='up', height=200, width=200)

    weight_file = st.file_uploader("Upload the model weight file", type=['pt', 'pth'])
    yaml_file = st.file_uploader("Upload the model YAML file", type=['yml'])
    arch_file = st.file_uploader("Upload the model architecture file", type=['py'])

    weight_file_path = None
    yaml_file_path = None
    arch_file_path = None

    if weight_file is not None:
        with open(weight_file.name, "wb") as f:
            f.write(weight_file.getbuffer())
        weight_file_path = weight_file.name

    if yaml_file is not None:
        with open(yaml_file.name, "wb") as f:
            f.write(yaml_file.getbuffer())
        yaml_file_path = yaml_file.name

    if arch_file is not None:
        with open(arch_file.name, "wb") as f:
            f.write(arch_file.getbuffer())
        arch_file_path = arch_file.name

    if st.button("Submit"):
        if weight_file and yaml_file and arch_file:

            weight_file = weight_file.name if hasattr(weight_file, 'name') else None
            yaml_file = yaml_file.name if hasattr(yaml_file, 'name') else None
            arch_file = arch_file.name if hasattr(arch_file, 'name') else None

            try:
                with open(yaml_file, mode='r', encoding='utf-8') as f:
                    x = yaml.safe_load(f)

                s = x.get('network_g', {}).get('type')  # s为模型的名字
                if s:
                    # 另存weight文件
                    weight_name = os.path.basename(weight_file_path)
                    yaml_name = os.path.basename(yaml_file_path)
                    arch_name = os.path.basename(arch_file_path)

                    new_weight_location = "LAM/ModelZoo/models"
                    new_weight_file_path = os.path.join(new_weight_location, weight_name)
                    new_weight_file_path = new_weight_file_path.replace("\\", "/")

                    with open(new_weight_file_path, 'wb') as f:
                        with open(weight_file, 'rb') as weight_file:
                            f.write(weight_file.read())
                    # 另存yaml文件

                    new_yaml_location = "LAM/ModelZoo/yaml"
                    new_yaml_file_path = os.path.join(new_yaml_location, yaml_name)
                    new_yaml_file_path = new_yaml_file_path.replace("\\", "/")
                    # st.write(new_yaml_file_path)

                    # 检查文件夹是否存在，如不存在则创建
                    os.makedirs(new_yaml_location, exist_ok=True)
                    # 从旧文件中读取数据
                    with open(yaml_file, 'r', encoding='utf-8') as ymal_file:
                        data = yaml.safe_load(ymal_file)
                    # 将数据写入新的 YAML 文件
                    with open(new_yaml_file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(data, f)

                    # 另存arch文件
                    new_arch_location = "LAM/ModelZoo/NN"
                    new_arch_file_path = os.path.join(new_arch_location, arch_name)
                    new_arch_file = new_arch_file_path.replace("\\", "/")

                    with open(new_arch_file, 'w', encoding='utf-8') as f:
                        with open(arch_file, 'r', encoding='utf-8') as arch_file:
                            f.write(arch_file.read())

                    if s not in session_state.model_names:
                        st.session_state.model_names = []
                        data = ModelData()
                        upload_result = data.update(new_weight_file_path, new_yaml_file_path, new_arch_file_path)
                        #print(type(upload_result))

                        st.session_state.model_names = upload_result[0]
                        st.session_state.MODEL_LIST = upload_result[1]
                        st.session_state.metrics = upload_result[2]

                        #st.write(st.session_state.metrics)

                    with st.container():
                        col1, col2 = st.columns([6, 4])
                        with col1:
                            st.markdown(" ")
                            st.markdown(" ")
                            st.markdown(" ")
                            st.markdown(" ")
                            gradient_text_html = f"""
                            <div style="
                                font-weight: bold;
                                background: -webkit-linear-gradient(left, red, orange);
                                background: linear-gradient(to right, red, orange);
                                -webkit-background-clip: text;
                                -webkit-text-fill-color: transparent;
                                display: inline;
                                font-size: 2em; /* 修改字体大小为2em */
                            ">
                            Successfully uploaded model \n named {s}
                            </div>
                            """

                            st.markdown(gradient_text_html, unsafe_allow_html=True)
                            #st.write(f"Successfully uploaded model named {s} ✅")

                        with col2:
                            lottie4 = load_lottiefile("Cartoon/star.json")
                            st_lottie(lottie4, key='great', height=250, width=250)

                    return True

            except Exception as e:
                st.error(f"Error occurred while processing YAML file: {e}")
        else:
            st.error("Please upload all necessary files and provide the architecture file path")
    return False
    #session_state.selected = st.selectbox("Select a model", session_state.model_names)


def display_selected_model():
    #session_state = initialize_session_state()

    if hasattr(st.session_state, 'model_names'):
        model_options = st.session_state.model_names
        model_pth = st.session_state.MODEL_LIST
        metrics = st.session_state.metrics
    else:
        session_state = initialize_session_state()
        model_options = session_state.model_names
        model_pth = session_state.MODEL_LIST
        metrics = session_state.metrics

    col1, col2 = st.columns([7.5, 2.5])
    # 在左侧列添加内容
    with col1:
        st.title('Performance Comparison')
        st.write(
            "👉 On this page, you can select different image super-resolution models for evaluation "
            "and see the visualization results of different indicators to help you analyze the advantages and disadvantages of the models. ✨")

    with col2:
        lottie9 = load_lottiefile("Cartoon/panda.json")
        st_lottie(lottie9, key='up', height=200, width=200)

    # st.write(session_state.model_names)
    #st.write("Please upload the images required for testing:")
    uploaded_file = st.file_uploader("Please upload the images required for testing", type=['jpg', 'png', 'jpeg'], key="uploader2")
    if uploaded_file is not None:
        st.image(uploaded_file)
        og = get_upload_img(uploaded_file)
        name = save_img(og)
        #st.markdown('Please select the model you want to compare. You can select multiple models.')

        # 用户自由选择模型名称
        selected_models = st.multiselect("Please select the models you want to compare( You can select multiple models).⭐️", model_options)
        number = len(selected_models)
        if number != 0:
            if st.button("Start"):
                st.write('Under evaluation. . .')
                model_names = []
                lam = []
                psnr = []
                ssim = []
                lpip = []
                DI = []
                for i in range(number):
                    model_name = selected_models[i]
                    model_names.append(model_name)
                    lam_result = click_lam(name, model_name, model_options, model_pth)
                    img_path = lam_result[0]
                    # di = lam_result[1]
                    # DI.append(di)
                    lam.append(img_path)
                    psnr.append(metrics[model_name]['psnr'])
                    ssim.append(metrics[model_name]['ssim'])
                    lpip.append(metrics[model_name]['lpips'])

                st.write("The comparison of average PSNR, SSIM, and LPIPS is as follows:")
                # 用户选择的模型和对应的指标值
                data = {"Model Name": model_names, "PSNR": ["{:.2f}".format(p) for p in psnr],
                        "SSIM": ssim,
                        "LPIPS": lpip}
                df = pd.DataFrame(data)
                df['PSNR'] = df['PSNR'].astype(float)
                df['SSIM'] = df['SSIM'].astype(float)
                df['LPIPS'] = df['LPIPS'].astype(float)

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # 绘制PSNR的柱状图
                ax[0].bar(df['Model Name'], df['PSNR'], color='#326fa8')
                ax[0].set_title('Model PSNR Comparison')
                ax[0].set_xlabel('Model Name')
                ax[0].set_ylabel('PSNR')

                # 标记最大值
                max_psnr = df['PSNR'].max()
                min_psnr = df['PSNR'].min()
                max_psnr_index = df['PSNR'].idxmax()
                ax[0].bar(df['Model Name'][max_psnr_index], max_psnr, color='#223d7d')

                # 设置Y轴范围
                ax[0].set_ylim([min_psnr - 1, max_psnr + 1])

                # 在条形上添加数值
                for i, v in enumerate(df['PSNR']):
                    ax[0].text(i, v + 0.01 * (max_psnr - min_psnr), "{:.2f}".format(v), ha='center', va='bottom')

                # 绘制SSIM的柱状图
                ax[1].bar(df['Model Name'], df['SSIM'], color='#dec50b')
                ax[1].set_title('Model SSIM Comparison')
                ax[1].set_xlabel('Model Name')
                ax[1].set_ylabel('SSIM')

                # 标记最大值
                max_ssim = df['SSIM'].max()
                min_ssim = df['SSIM'].min()
                max_ssim_index = df['SSIM'].idxmax()
                ax[1].bar(df['Model Name'][max_ssim_index], max_ssim, color='#e69809')

                # 设置Y轴范围
                ax[1].set_ylim([min_ssim - 0.005, max_ssim + 0.005])

                # 在条形上添加数值
                for i, v in enumerate(df['SSIM']):
                    ax[1].text(i, v + 0.005 * (max_ssim - min_ssim), "{:.4f}".format(v), ha='center',
                               va='bottom')

                st.pyplot(fig)
                fig.tight_layout(pad=3.0)

                # 如需要绘制LPIPS图，可以创建新的子图
                fig2, ax2 = plt.subplots(figsize=(6, 5))

                # 绘制LPIPS的折线图
                ax2.plot(df['Model Name'], df['LPIPS'], marker='o', color='#32a88c')
                ax2.set_title('Model LPIPS Comparison')
                ax2.set_xlabel('Model Name')
                ax2.set_ylabel('LPIPS')

                min_lpip = df['LPIPS'].min()
                min_lpip_index = df['LPIPS'].idxmin()
                max_lpip = df['LPIPS'].max()
                ax2.plot(df['Model Name'][min_lpip_index], min_lpip, marker='o', color='#134a16')

                # 在折线图上的每个数据点上添加数值
                for i, v in enumerate(df['LPIPS']):
                    ax2.text(i, v + 0.05 * (max_lpip - min_lpip), "{:.4f}".format(v), ha='center', va='bottom')

                # 使用streamlit显示图形
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig2)
                    with col2:
                        lottie3 = load_lottiefile("Cartoon/evaluation.json")
                        st_lottie(lottie3, key='eval', height=360, width=360)

                # st.table(df)

                st.write('The comparison results of the LAM diagram are as follows:')
                # 图片恢复后储存到这里
                for i in range(len(lam)):
                    st.write(f"Model Name: {model_names[i]}")
                    st.image(lam[i], caption='', use_column_width=True)
                    # st.write(f"The DI of this case is {DI[i]:.2f}")
                    remove_file(lam[i])

if selected=="Introduce":
    #Header

    st.title('Welcome to PixelMagic! 👋')
    st.subheader('*A new super-resolution image restoration and model performance evaluation tool.*')
    st.divider() #一个*为斜体，两个不是
    # Use Cases
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header('Use Cases')
            st.markdown(
                """
                - _Are you tired of low-resolution images and looking to enhance your image quality with super-resolution restoration tools?_
                - _Interested in exploring the capabilities of super-resolution image restoration to improve the visual quality of your images?_
                - _Conducting performance evaluations of various models for super-resolution image restoration to fine-tune your image processing workflows?_
                - Just here to delve into the world of image enhancement and learn more about the fascinating domain of super-resolution image restoration tools? 
                """
            )
        with col2:
            lottie2 = load_lottiefile("Cartoon/robot.json")
            st_lottie(lottie2, key='place', height=370, width=350)

    st.divider()

    # Tutorial Video教程视频
    st.header('Tutorial Video')
    # video_file = open('Similo_Tutorial3_compressed.mp4', 'rb')
    # video_bytes = video_file.read()
    # st.video(video_bytes)


if selected == "Image Restoration":
    two_page()

if selected == 'Upload':
    update_session_state()

#Performance Comparison Page
if selected=='Perf Comp':
    display_selected_model()

