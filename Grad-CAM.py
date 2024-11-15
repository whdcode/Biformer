"""
Conv-BiFormer-model
From
"author: Huidong Wu
github: https://github.com/whdcode
email: 1505261567@qq.com"
”
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import nibabel as nib
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from convformer5 import Convformer
from utils import GradCAM, show_cam_on_image, create_brain_mask, apply_mask
np.seterr(divide='ignore', invalid='ignore')

def AvgImagesCam(images_fold_path, cam):
    global avg_cam
    img_list = os.listdir(images_fold_path)
    img_cam_list = []
    assert os.path.exists(images_fold_path), "file: '{}' dose not exist.".format(images_fold_path)
    for i, img in enumerate(img_list):
        img_path = os.path.join(images_fold_path, img)
        MRI_array = np.load(img_path, allow_pickle=True)
        MRI_array = MRI_array[0][0]  # 数据结构为 [[data， label]]
        input_tensor = torch.FloatTensor(MRI_array).unsqueeze(0).unsqueeze(0).cuda()

        grayscale_cam = cam(input_tensor=input_tensor, target_category=0)[0]
        img_cam_list.append(grayscale_cam)
    img_cam_list = np.array(img_cam_list)
    avg_cam = img_cam_list.mean(axis=0)
    avg_cam = np.power(avg_cam, 0.5)
    return avg_cam, MRI_array


class ReshapeTransform:
    def __init__(self, model):
        self.h = 6
        self.w = 6
        self.d = 6

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 2:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     self.d,
                                     x.size(2))

        result = result.permute(0, 4, 1, 2, 3)
        return result


def main():
    # 模型实例化
    global grayscale_cam, img_name
    model = Convformer(num_classes=1).cuda()
    # 使用五折模型测试效果最好的折模型权重
    # 权重路径
    weights_path = "D:\\translate\\MVFP-Net\\MVFP-Net\\comvfomer2_3\\ablation_weights_MCI\\6_MVFF+CMSF+CN+VITE" \
                   "+MAC-fold6.pth"
    # 载入训练号的模型权重
    model.load_state_dict(torch.load(weights_path, map_location="cpu")["model_state_dict"], strict=False)
    # 目标映射网络层，可以使用多个层，最终取热力值平均值
    target_layers = [model.backbone.DS3 ]
    # 文件夹名称（可视化层的名字，最后是以点相隔的层名字组成）
    target_layers_name = "model.backbone.DS3_block23_gam0.5_mul1.1_DATA86"
    # 如果文件夹名称里包含了blocks，说明这是需要进行序列到特征图的重塑转换，不包含说明目标层的数据流为特征图形式
    if "blocks" not in target_layers_name:
        trans = None
    else:
        trans = ReshapeTransform(model)
    # 实例化gardcam
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=True,
                  reshape_transform=trans)
    # 可以选择五折中最好的那一折
    fold_num = 3
    # 需要可视化的sMRI截面数量
    num_slices = 60
    # 获取截面的开始位置（如冠状面的厚度为112，start = 25表示从第25个冠状截面开始进行可视化
    start = 25
    # 获取截面的间隔，以体素间隔为单位，默认为0
    space_num = 0
    # 决定热力值范围下限
    atten_area_rate = 0.0  # Coronal Plane ：0.67 or Axial Plane:0.67 or Sagittal Plane:0.63
    # 选择需要可视化的病症类别
    img_class = 'pMCI'
    # sMRI影像文件夹路径
    img_root = f'D:\\translate\\MVFP-Net\\MVFP-Net\\visitual_explaintion\\visitual_test_fold3\\MCI\\AVG_CAM'  # ADHC or MCI
    # 这里任选一张sMRI为了后续生成去除背景的mask,或者作为AVGcam覆盖的模板，随便一张如'005_S_0546_1.npy'
    MRI_mask = np.load(os.path.join(img_root, '005_S_0546_1.npy'), allow_pickle=True)[0][0]
    MRI_temp = None
    # 对单张影像进行可视化S还是平均热激活A可视化
    SinOrAvg = 'A'
    if SinOrAvg == 'S':
        # 目标sMRI影像的名称
        img_name = '005_S_0546_1.npy'
        sin_img_path = os.path.join(img_root, img_name)
        assert os.path.exists(sin_img_path), "file: '{}' dose not exist.".format(sin_img_path)
        # [[data， label]] 取出数据data
        sin_img_array = np.load(sin_img_path, allow_pickle=True)[0][0]
        input_tensor = torch.FloatTensor(sin_img_array).unsqueeze(0).unsqueeze(0).cuda()
        grayscale_cam = cam(input_tensor=input_tensor, target_category=0)[0, :]
        MRI_temp = sin_img_array
    elif SinOrAvg == 'A':
        grayscale_cam, _ = AvgImagesCam(img_root, cam)
        MRI_temp = MRI_mask

    # 以下为了去除无关背景
    # 创建一个Nifti1Image对象, 得到由112x112x112的同样空间尺寸的映射目标
    nifti_image = nib.Nifti1Image(MRI_mask, affine=np.eye(4))
    sMRI_mask = nifti_image.get_fdata()
    # 进行背景去除
    brain_mask = create_brain_mask(sMRI_mask)
    masked_heatmap = apply_mask(grayscale_cam, brain_mask)
    # 再次遮罩聚焦
    masked_heatmap = masked_heatmap / np.max(masked_heatmap) + 1e-12
    # 控制热力值下限
    masked_heatmap1 = create_brain_mask(masked_heatmap, atten_area_rate)
    masked_heatmap = apply_mask(masked_heatmap, masked_heatmap1)
    for i in range(3):
        if i == 0:
            n = 'Axial Plane'
        elif i == 1:
            n = "Sagittal Plane"
        elif i == 2:
            n = 'Coronal Plane'

        # 可视化结果的保存路径
        if SinOrAvg == 'S':
            name_images_path = os.path.join(f'./new_vis/fold{fold_num}_{img_class}', img_name[0:12],
                                            target_layers_name, n)
            if not os.path.exists(name_images_path):
                os.makedirs(name_images_path)
        elif SinOrAvg == 'A':
            name_images_path = os.path.join(f'./new_vis/fold{fold_num}_{img_class}', 'AVG_CAM',
                                            target_layers_name, n)
            if not os.path.exists(name_images_path):
                os.makedirs(name_images_path)

        # 切片的方式，隔多少体素进行一次切片采集
        for k in range(num_slices):
            if space_num:
                f = start + (space_num + 1) * k
            else:
                f = start + k
            # 对应视角的切片数据和热图
            if i == 0:
                MRI_img = MRI_temp[:, :, f]
                grayscale_cam = masked_heatmap[:, :, f]
            elif i == 1:
                MRI_img = MRI_temp[f, :, :]
                grayscale_cam = masked_heatmap[f, :, :]
            elif i == 2:
                MRI_img = MRI_temp[:, f, :]
                grayscale_cam = masked_heatmap[:, f, :]
            # 需要将灰度图片转为RGB
            img = cv2.cvtColor(MRI_img, cv2.COLOR_GRAY2BGR)
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            plt.subplot(1, 2, 1)
            plt.imshow(np.rot90(MRI_img, 1), cmap='gray')
            plt.title('MRI Slice {}'.format(f))
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(np.rot90(visualization, 1))
            plt.title('Heatmap {}'.format(f))
            plt.axis('off')

            # plt.subplot(1, 2, 1)
            # # visualization = np.where(visualization < 52, 52, visualization)
            # heat = plt.imshow(visualization, cmap='jet', vmin=0, vmax=1)
            # cax = plt.axes([0.91, 0.2, 0.035, 0.7])
            # cbar = plt.colorbar(heat, ax=plt.gca(), cax=cax)
            # cbar.set_ticks([0.2, 1])  # 设置热力棒的刻度
            # cbar.ax.set_position([0.71, 0.2, 0.045, 0.7])

            plt.tight_layout()
            plt.savefig(f'{name_images_path}/grad-CAM{f}.png', dpi=300)
            plt.show()


if __name__ == '__main__':
    main()
