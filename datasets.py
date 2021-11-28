# -*- encoding: utf-8 -*-
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
from datetime import datetime

import pandas
import torch
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def main(name):
    print(f'Hi, {name}', datetime.now())
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)


class MnistDataset(Dataset):
    """数据集的管理"""

    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        # 确定图像的标签
        label = self.data_df.iloc[item, 0]
        target = torch.zeros(10)
        target[label] = 1.0

        # 归一化图像数据，从 0~255 到 0~1，从整数到浮点
        image_values = torch.FloatTensor(self.data_df.iloc[item, 1:].values) / 255.0

        return label, image_values, target

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label= ".format(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')