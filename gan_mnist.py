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
import matplotlib.pyplot as plt
from datetime import datetime


def main(name):
    print(f'Hi, {name}', datetime.now())

    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    plt.show()
