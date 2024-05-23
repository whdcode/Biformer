import os
# import torchio
# from torchio import *
# import volumentations
# from volumentations import *
import torch
from torch.utils.data import Dataset, DataLoader
from dataset_aug import *


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, transform=None):
        self.data = data_root
        self.label = data_label
        self.transform = transform

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]

        if self.transform is not None:
            data = self.transform(data)

        return torch.tensor(data).float(), torch.tensor(labels).float()

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


class CustomDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.data_files = os.listdir(self.data_dir)
        self.transform = transforms.Compose([
            xyz_rotate(-10, 10, rate=0.5),  # 0.2 0.5
            flip(rate=0.5), # 0.2 0.5
            mask(rate=0.5, mask_nums=2, intersect=False),   # 0.2 0.5
            #equa_hist(),
            RandomCrop3D((80, 80, 80), radio=0.5),  # 0.2 0.4
            contrast(),
        ])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.data_files[idx]), allow_pickle=True)
        image = data[0][0]
        label = data[0][1]

        if self.split == 'train':
            image = self.transform(image)
        return torch.tensor(image.copy()).float().unsqueeze(0), torch.tensor(label).float()

class noisy(object):
    def __init__(self, radio, probability):
        self.radio = radio
        self.probability = probability

    def __call__(self, data):
        if np.random.uniform(0, 1) > self.probability:
            return data

        l, w, h = data.shape
        num = int(l * w * h * self.radio)
        for _ in range(num):
            x = np.random.randint(0, l)
            y = np.random.randint(0, w)
            z = np.random.randint(0, h)
            noise = np.random.uniform(0, self.radio)
            data[x, y, z] = data[x, y, z] + noise
        return data




