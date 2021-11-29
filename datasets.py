"""
=================================================
@path   : gan -> datasets.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-28 10:29
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
import random
from datetime import datetime

import h5py
import numpy
import pandas
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """CelebA 数据集"""

    def __init__(self, hdf5_file):
        self.file_object = h5py.File(hdf5_file, 'r')
        self.dataset = self.file_object['img_align_celeba']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if item >= len(self.dataset):
            raise IndexError()
        img = numpy.array(self.dataset[str(item) + '.jpg'])
        img = crop_center(img, 128, 128)
        return torch.cuda.FloatTensor(img).permute(2, 0, 1).view(1, 3, 128, 128) / 255.0

    def plot_image(self, index):
        img = numpy.array(self.dataset[str(index) + '.jpg'])
        img = crop_center(img)
        plt.imshow(img, interpolation='nearest')


class MnistDataset(Dataset):
    """MNIST 数据集"""

    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 确定图像的标签
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0

        # 归一化图像数据，从 0~255 到 0~1，从整数到浮点
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0

        return label, image_values, target

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label= ".format(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')


def test_mnist_data():
    df = pandas.read_csv("datasets/mnist_train.csv", header=None)
    print(df.head())
    print(df.info())
    # 从 pandas 的 dataframe 中获取数据
    row = 13
    row_data = df.iloc[row]
    label = row_data[0]
    img = row_data[1:].values.reshape(28, 28)
    plt.title(("label={}".format(label)))
    plt.imshow(img, interpolation='none', cmap='Blues')
    plt.show()


def generate_real():
    real_data = torch.FloatTensor([
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2),
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2)
    ])
    return real_data


def generate_random(size):
    """随机生成均匀分布的数据"""
    random_data = torch.rand(size)
    return random_data


def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


def crop_center(img, new_width, new_height):
    """输入给定的大小，基于中心剪裁"""
    height, width, _ = img.shape
    startx = width // 2 - new_width // 2
    starty = height // 2 - new_height // 2
    return img[starty:starty + new_height, startx:startx + new_width]


def test_MnistDataset():
    mnist_dataset.plot_image(17)
    pass


def test_CelebADataset():
    celeba_dataset.plot_image(43)
    pass


def main(name):
    print(f'Hi, {name}', datetime.now())
    # 测试 MNIST 的数据库
    # test_mnist_data()
    # test_MnistDataset()
    test_CelebADataset()
    pass


mnist_dataset = MnistDataset('datasets/mnist_train.csv')
mnist_test_dataset = MnistDataset('datasets/mnist_test.csv')
celeba_dataset = CelebADataset('datasets/celeba_aligned_small.h5py')

if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    plt.show()
