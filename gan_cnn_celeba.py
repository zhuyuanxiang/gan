"""
=================================================
@path   : gan -> gan_cnn_celeba.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-29 14:55
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from datetime import datetime

import matplotlib.pyplot as plt
import pandas
import random
import torch
import torch.nn as nn

from datasets import celeba_dataset
from datasets import generate_random_image
from datasets import generate_random_seed
from gan_simple import Discriminator
from gan_simple import Generator


class CNNDiscriminator(Discriminator):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=(8, 8), stride=(2, 2)),
                nn.BatchNorm2d(256),
                # nn.LeakyReLU(0.2),
                nn.GELU(),

                nn.Conv2d(256, 256, kernel_size=(8, 8), stride=(2, 2)),
                nn.BatchNorm2d(256),
                # nn.LeakyReLU(0.2),
                nn.GELU(),

                nn.Conv2d(256, 3, kernel_size=(8, 8), stride=(2, 2)),
                # nn.LeakyReLU(0.2),
                nn.GELU(),

                nn.Flatten(0, -1),     # 从第 0 个维度开始将所有维度摊平
                nn.Linear(3 * 10 * 10, 1),
                nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


class CNNGenerator(Generator):
    def __init__(self):
        super(CNNGenerator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(100, 3 * 11 * 11),
                # nn.LeakyReLU(0.2),
                nn.GELU(),

                nn.Unflatten(0, (1, 3, 11, 11)),

                nn.ConvTranspose2d(3, 256, kernel_size=(8, 8), stride=(2, 2)),
                nn.BatchNorm2d(256),
                # nn.LeakyReLU(0.2),
                nn.GELU(),

                nn.ConvTranspose2d(256, 256, kernel_size=(8, 8), stride=(2, 2)),
                nn.BatchNorm2d(256),
                # nn.LeakyReLU(0.2),
                nn.GELU(),

                nn.ConvTranspose2d(256, 3, kernel_size=(8, 8), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(3),

                nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


def main(name):
    print(f'Hi, {name}', datetime.now())
    # test_discriminator()
    # test_generator()
    train_GAN()
    pass


def train_GAN():
    # 训练 GAN
    D = CNNDiscriminator().to(device)
    G = CNNGenerator().to(device)

    epochs = 1
    for epoch in range(epochs):
        print(f"--->第{epoch}次迭代<---")
        for image_data_tensor in celeba_dataset:
            D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
            D.train(G.forward(generate_random_seed(100)).detach(), torch.cuda.FloatTensor([0.0]))
            G.train(D, generate_random_seed(100), torch.cuda.FloatTensor([1.0]))
        D.plot_progress()
        G.plot_progress()

    # 推理 Generator
    f, axarr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(100))
            img = output.detach().permute(0, 2, 3, 1).view(128, 128, 3).cpu().numpy()
            axarr[i, j].imshow(img, interpolation='none', cmap='Blues')


def test_generator():
    G = CNNGenerator().to(device)
    output = G.forward(generate_random_seed(100))
    img = output.detach().permute(0, 2, 3, 1).view(128, 128, 3).cpu().numpy()
    plt.imshow(img, interpolation='none', cmap='Blues')


def test_discriminator():
    D = CNNDiscriminator().to(device)
    for image_data_tensor in celeba_dataset:
        D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
        D.train(generate_random_image((1, 3, 128, 128)).to(device), torch.cuda.FloatTensor([0.0]))
    D.plot_progress()
    for i in range(4):
        image_data_tensor = celeba_dataset[random.randint(0, 20000)]
        print(D.forward(image_data_tensor).item())
    for i in range(4):
        print(D.forward(generate_random_image((1, 3, 128, 128))).item())


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("using cuda:", torch.cuda.get_device_name())
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(__author__)
    plt.show()
