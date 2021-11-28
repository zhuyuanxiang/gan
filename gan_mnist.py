# -*- encoding: utf-8 -*-
import torch
import torch
from abc import ABCMeta
from abc import abstractmethod

from torch.nn import Module

"""
=================================================
@path   : gan -> gan_mnist.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-24 16:09
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas
import matplotlib.pyplot as plt
from datetime import datetime
from tools import func_time


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


class Classifier(Module, metaclass=ABCMeta):
    """用于分类"""

    def __init__(self):
        super(Classifier, self).__init__()

        self.model = self.create_network()

        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []
        pass

    @abstractmethod
    def create_network(self):
        pass

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
            pass
        if self.counter % 10000 == 0:
            print("counter=", self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass


class LinearClassifier(Classifier):
    def create_network(self):
        return nn.Sequential(
                nn.Linear(784, 200),
                nn.Sigmoid(),
                nn.Linear(200, 10),
                nn.Sigmoid()
        )


class LeakyClassifier(Classifier):

    def create_network(self):
        return nn.Sequential(
                nn.Linear(784, 200),
                nn.LeakyReLU(0.02),
                nn.Linear(200, 10),
                nn.Sigmoid()
        )


class LayerNormLinearClassifier(Classifier):
    def create_network(self):
        return nn.Sequential(
                nn.Linear(784, 200),
                nn.Sigmoid(),
                nn.LayerNorm(200),
                nn.Linear(200, 10),
                nn.Sigmoid()
        )


class BCELinearClassifier(LinearClassifier):
    def __init__(self):
        super(BCELinearClassifier, self).__init__()
        self.loss_function = nn.BCELoss()


class BCELeakyClassifier(LeakyClassifier):
    def __init__(self):
        super(BCELeakyClassifier, self).__init__()


class AdamLayerNormLinearClassifier(LayerNormLinearClassifier):
    def __init__(self):
        super(AdamLayerNormLinearClassifier, self).__init__()
        self.optimiser = torch.optim.Adam(self.parameters())


def main(name):
    print(f'Hi, {name}', datetime.now())
    # 测试 MNIST 的数据库
    # mnist_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    mnist_dataset = MnistDataset('datasets/mnist_train.csv')
    mnist_test_dataset = MnistDataset('datasets/mnist_test.csv')

    # test_mnist_class(mnist_dataset)
    # test_mnist_class(mnist_test_dataset)

    # C = LinearClassifier()
    # train_network(C, mnist_dataset, mnist_test_dataset)
    # test_network(C, mnist_test_dataset)

    # C = LeakyClassifier()
    C = AdamLayerNormLinearClassifier().to(device)
    train_network(C, mnist_dataset, device)
    test_network(C, mnist_test_dataset, device)
    pass


@func_time
def test_network(C, mnist_test_dataset, device):
    record = 19
    print("label=", mnist_test_dataset[record][0])
    image_data = mnist_test_dataset[record][1]
    output = C.forward(image_data.to(device))
    pandas.DataFrame(output.detach().cpu().numpy()).plot(kind='bar', legend=False, ylim=(0, 1))

    score, items = 0, 0
    for label, image_data_tensor, target_tensor in mnist_test_dataset:
        answer = C.forward(image_data_tensor.to(device)).detach().cpu().numpy()
        if answer.argmax() == label:
            score += 1
        items += 1
    print("正确标记数据=", score)
    print("原始测试数据=", items)
    print("精确度=", score / items)


@func_time
def train_network(C, mnist_dataset, device):
    epochs = 4
    for i in range(epochs):
        print("training epoch", i + 1, "of", epochs)
        for label, image_data_tensor, target_tensor in mnist_dataset:
            C.train(image_data_tensor.to(device), target_tensor.to(device))
            pass
        pass
    C.plot_progress()


def test_mnist_class():
    mnist_dataset = MnistDataset('datasets/mnist_train.csv')
    print(mnist_dataset[100])
    mnist_dataset.plot_image(9)


def mnist_data():
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


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    plt.show()
