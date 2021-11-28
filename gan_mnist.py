# -*- encoding: utf-8 -*-

from datasets import MnistDataset
from linear_classifier import AdamLayerNormLinearClassifier
from linear_classifier import test_network
from linear_classifier import train_network

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
import pandas
import matplotlib.pyplot as plt
from datetime import datetime


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
