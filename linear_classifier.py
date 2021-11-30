# -*- encoding: utf-8 -*-
from tools import get_device

"""
=================================================
@path   : gan -> linear_classifier.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-28 10:30
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from abc import ABCMeta
from abc import abstractmethod
from datetime import datetime

import pandas
import torch
from torch import nn as nn
from torch.nn import Module

from datasets import mnist_dataset
from datasets import mnist_test_dataset
from tools import func_time


def main(name):
    print(f'Hi, {name}', datetime.now())

    # test_mnist_class(mnist_dataset)
    # test_mnist_class(mnist_test_dataset)

    # C = LinearClassifier()
    # C = LeakyClassifier()
    C = AdamLayerNormLinearClassifier()
    train_network(C, mnist_dataset)
    test_network(C, mnist_test_dataset)
    pass


class Classifier(Module, metaclass=ABCMeta):
    """用于分类"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.model = self.create_network()  # 抽象函数，继承时需要实现
        self.loss_function = self.create_loss_func()  # 默认为 MSELoss()
        self.optimiser = self.create_optimiser()  # 默认为 SGD()
        self.counter = 0
        self.progress = []
        pass

    def create_optimiser(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def create_loss_func(self):
        return nn.MSELoss()

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

        # 梯度归零，反向传播，更新权重
        self.optimiser.zero_grad()  # 将计算图中的梯度全部归0，即参数初始化
        loss.backward()  # 从损失函数中反向传播梯度
        self.optimiser.step()  # 使用梯度更新网络参数
        # 注1：在每次训练之前都需要将梯度归零，否则梯度会累加
        # 注2：使用每次反向传播得到的梯度更新网络的参数
        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass


class LinearClassifier(Classifier):
    """原始的线性分类器"""

    def create_network(self):
        return nn.Sequential(
                nn.Linear(784, 200),
                nn.Sigmoid(),
                nn.Linear(200, 10),
                nn.Sigmoid()
        )


class LeakyClassifier(Classifier):
    """使用 LeakyReLU() 损失函数"""

    def create_network(self):
        return nn.Sequential(
                nn.Linear(784, 200),
                nn.LeakyReLU(0.02),
                nn.Linear(200, 10),
                nn.Sigmoid()
        )


class LayerNormLinearClassifier(Classifier):
    """使用层归一化 LayerNorm()"""

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

    def create_loss_func(self):
        return nn.BCELoss()


class BCELeakyClassifier(LeakyClassifier):
    def __init__(self):
        super(BCELeakyClassifier, self).__init__()

    def create_loss_func(self):
        return nn.BCELoss()


class AdamLayerNormLinearClassifier(LayerNormLinearClassifier):
    def __init__(self):
        super(AdamLayerNormLinearClassifier, self).__init__()

    def create_optimiser(self):
        return torch.optim.Adam(self.parameters())


@func_time
def test_network(C, test_dataset):
    record = 19
    print("label=", test_dataset[record][0])
    image_data = test_dataset[record][1]
    output = C.forward(image_data.to(device))
    pandas.DataFrame(output.detach().cpu().numpy()).plot(kind='bar', legend=False, ylim=(0, 1))

    score, items = 0, 0
    for label, image_data_tensor, target_tensor in test_dataset:
        answer = C.forward(image_data_tensor.to(device)).detach().cpu().numpy()
        if answer.argmax() == label:
            score += 1
        items += 1
    print("正确标记数据=", score)
    print("原始测试数据=", items)
    print("精确度=", score / items)


@func_time
def train_network(C, train_dataset):
    epochs = 4
    for i in range(epochs):
        print("training epoch", i + 1, "of", epochs)
        for label, image_data_tensor, target_tensor in train_dataset:
            C.train(image_data_tensor.to(device), target_tensor.to(device))
            pass
        pass
    C.plot_progress()


def test_mnist_class(dataset):
    print(dataset[100])
    dataset.plot_image(9)


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    device = get_device()
