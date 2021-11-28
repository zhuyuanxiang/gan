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
import pandas, numpy, random
from gan_simple import Discriminator
from gan_simple import Generator
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import generate_random, mnist_dataset


class GANDiscriminator(Discriminator):
    def __init__(self):
        super(GANDiscriminator, self).__init__()
        # 重新设置模型后，必须重新设置优化器，否则就使用了父模型的参数（这点我其实挺迷惑的，又不报错，但是结果不对）
        self.model = nn.Sequential(
                nn.Linear(784, 200),
                nn.Sigmoid(),
                nn.Linear(200, 1),
                nn.Sigmoid()
        )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)


def test_discriminator():
    D = GANDiscriminator()
    for label, image_data_tensor, target_tensor in mnist_dataset:
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
        D.train(generate_random(784), torch.FloatTensor([0.0]))
        pass
    D.plot_progress()

    print("--->对真实数据的判断结果<---")
    for i in range(4):
        image_data_tensor = mnist_dataset[random.randint(0, 60000)][1]
        print(D.forward(image_data_tensor).item())
        pass

    print("--->对伪装数据的判断结果<---")
    for i in range(4):
        image_fake_tensor = generate_random(784)
        print(D.forward(image_fake_tensor).item())
        pass
    pass


class GANGenerator(Generator):
    def __init__(self):
        super(GANGenerator, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(1, 200),
                nn.Sigmoid(),
                nn.Linear(200, 784),
                nn.Sigmoid()
        )

        pass


def main(name):
    print(f'Hi, {name}', datetime.now())
    test_discriminator()
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    plt.show()
