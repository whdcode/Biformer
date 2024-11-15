import os.path

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Dataset import CustomDataset
from convformer5 import Convformer
from torch.utils import data as torch_data

from utils import _init_fn


class T_sne_visual():
    def __init__(self, model, dataset, dataloader, output_path, target_dict):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.output_path = output_path
        self.target_dict = target_dict
        self.ori = 0
        self.class_list = []

    def visual_dataset(self):
        imgs = []
        labels = []
        self.ori = 1
        save_path = os.path.join(output_path, "Original_dataset_visualize_result")
        for img, label in self.dataset:
            imgs.append(np.array(img).transpose((3, 2, 1, 0)).reshape(-1))
            tag = label.item()
            labels.append(tag)
        imgs = np.array(imgs)
        self.t_sne(imgs, labels, title=f'Original dataset visualize result', save_path=save_path, ori=self.ori)

    def visual_feature_map(self, layer):
        self.ori = 0
        save_path = os.path.join(output_path, f'{layer}_feature_map')
        self.model.eval()
        with torch.no_grad():
            self.feature_map_list = []
            labels = []
            getattr(self.model, layer).DS5.register_forward_hook(self.forward_hook)
            for img, label in self.dataloader:
                img = img.cuda()
                self.model(img)
                for i in label.tolist():
                    # tag = self.class_list[i]
                    labels.append(i)
            self.feature_map_list = torch.cat(self.feature_map_list, dim=0)
            self.feature_map_list = torch.flatten(self.feature_map_list, start_dim=1)
            feature_map_array = np.array(self.feature_map_list.cpu())
            # pca = PCA(n_components=2)
            # feature_map_array = pca.fit_transform(feature_map_array)
            # feature_map_array = (feature_map_array - np.mean(feature_map_array, axis=0)) / np.std(feature_map_array,
            #                                                                                       axis=0)
            self.t_sne(feature_map_array, np.array(labels), title=f'After {layer} feature map',
                       save_path=save_path, ori=self.ori)

    def forward_hook(self, model, input, output):
        self.feature_map_list.append(output)

    def set_plt(self, start_time, end_time, title):
        plt.title(f'{title}')
        print(f"time consume:{end_time - start_time:.3f} s")
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

    def t_sne(self, data, label, title, save_path, ori):
        # t-sne处理
        print('starting T-SNE process')
        start_time = time()
        data = TSNE(n_components=2, init='pca', perplexity=40, learning_rate=220,random_state=42).fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
        df.insert(loc=1, column='label', value=label)
        end_time = time()
        print('Finished')

        # # Fit a logistic regression classifier
        # clf = LogisticRegression().fit(df[['x', 'y']], df['label'])

        # Create a mesh to plot the decision boundary
        h = .02  # step size in the mesh
        x_min, x_max = df['x'].min() - 0.1, df['x'].max() + 0.1
        y_min, y_max = df['y'].min() - 0.1, df['y'].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # # Predict class using the trained classifier
        # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)

        # 绘图
        # 此处绘制散点图
        sns.scatterplot(x='x', y='y', hue='label', style='label', s=18, palette="Set2", markers=["o", "v"], data=df,
                        legend=False)

        # 绘制决策线
        # if not ori:
        #     plt.contour(xx, yy, Z, levels=[0.5], colors='grey', linestyles='dashed', alpha=0.4, linewidths=0.8)

        # 类别表 {1: 'AD', 0: 'HC'} or {1: 'pMCI', 0: 'sMCI'}
        if self.target_dict:
            label_mapping = {1: 'AD', 0: 'NC'}
        else:
            label_mapping = {1: 'pMCI', 0: 'sMCI'}

        unique_labels = df['label'].unique()
        legend_labels = [label_mapping[label] for label in unique_labels]
        palette = sns.color_palette("Set2")

        # Create custom legend markers
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10, linestyle='None')
                   for i in range(len(unique_labels))]

        # 修改标记样式
        for i, marker in enumerate(["o", "v"]):
            handles[i].set_marker(marker)

        # Set the legend
        plt.legend(handles, legend_labels, loc="upper right", fontsize=10)
        self.set_plt(start_time, end_time, title)
        plt.savefig(save_path, dpi=400)
        plt.show()

if __name__ == '__main__':
    Target_class = 'ADvsHC'  # ADvsHC or sMCIvspMCI
    fold_num = 5
    for fold_num in range(1):
        if Target_class == 'ADvsHC':
            img_root = f'D:\\translate\\MVFP-Net\\MVFP-Net\\visitual_explaintion\\visitual_test_fold3\\ADHC'
            target_dict = 1
        else:
            img_root = f'D:\\translate\\MVFP-Net\\MVFP-Net\\visitual_explaintion\\visitual_test_fold3\\MCI'
            target_dict = 0
        imgset = CustomDataset(img_root, split='val')
        output_path = f"ADHCxperformance_curvecon52/T-SNE/{Target_class}/f{fold_num+1}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_loader = torch_data.DataLoader(
            imgset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=_init_fn
        )
        # 载入预训练权重
        checkpoint = torch.load(f"D:\\translate\\MVFP-Net\\MVFP-Net\\comvfomer2_3\\ablation_weights_ADHC\\6_MVFF+CMSF+CN+VITE+MAC-fold5.pth", map_location='cpu')
        net = Convformer().cuda()
        net.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # 实例化T-SNE
        t = T_sne_visual(net, imgset, img_loader, output_path, target_dict)
        # 绘制原数据集T-SNE分布
        t.visual_dataset()
        # 绘制模型经过训练后的区分效果的T-SNE分布
        t.visual_feature_map('backbone')
