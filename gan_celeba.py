"""
=================================================
@path   : gan -> gan_celeba.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-29 10:32
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
import random
from datetime import datetime

import matplotlib.pyplot as plt
import pandas
import torch
import torch.nn as nn

from datasets import celeba_dataset
from datasets import generate_random_image
from datasets import generate_random_seed
from gan_simple import Discriminator
from gan_simple import Generator


class CelebADiscriminator(Discriminator):
    def __init__(self):
        super(CelebADiscriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Flatten(0, -1),  # 从 0 维摊平
                nn.Linear(3 * 128 * 128, 100),
                nn.LeakyReLU(0.02),
                nn.LayerNorm(100),
                nn.Linear(100, 1),
                nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


class CelebAGenerator(Generator):
    def __init__(self):
        super(CelebAGenerator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(100, 3 * 10 * 10),
                nn.LeakyReLU(0.02),
                nn.LayerNorm(3 * 10 * 10),
                nn.Linear(3 * 10 * 10, 3 * 128 * 128),
                nn.Sigmoid(),
                nn.Unflatten(0, (3, 128, 128))  # 从 0 维重组
        )
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


def main(name):
    print(f'Hi, {name}', datetime.now())
    # test_celeba_discriminator()
    # test_celeba_generator()
    test_GAN()
    pass


def test_GAN():
    D = CelebADiscriminator().to(device)
    G = CelebAGenerator().to(device)
    epochs = 4
    for epoch in range(epochs):
        print(f"--->第{epoch}次迭代<---")
        for image_data_tensor in celeba_dataset:
            D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
            # generate_data_tensor = generate_random_seed(100)
            # D.train(G.forward(generate_data_tensor).detach(), torch.cuda.FloatTensor([0.0]))
            # G.train(D, generate_data_tensor, torch.cuda.FloatTensor([1.0]))
            D.train(G.forward(generate_random_seed(100)).detach(), torch.cuda.FloatTensor([0.0]))
            G.train(D, generate_random_seed(100), torch.cuda.FloatTensor([1.0]))
    D.plot_progress()
    G.plot_progress()
    f, axarr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(100))
            img = output.detach().permute(1, 2, 0).cpu().numpy()
            axarr[i, j].imshow(img, interpolation='none', cmap='Blues')


def test_celeba_generator():
    G = CelebAGenerator().to(device)
    output = G.forward(generate_random_seed(100))
    img = output.detach().cpu().numpy()
    plt.imshow(img, interpolation='none', cmap='Blues')


def test_celeba_discriminator():
    D = CelebADiscriminator().to(device)
    for image_data_tensor in celeba_dataset:
        D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
        D.train(generate_random_image((128, 128, 3)), torch.cuda.FloatTensor([0.0]))
    D.plot_progress()
    for i in range(4):
        image_data_tensor = celeba_dataset[random.randint(0, 20000)]
        print(D.forward(image_data_tensor).item())
    for i in range(4):
        print(D.forward(generate_random_image((218, 178, 3))).item())


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("using cuda:", torch.cuda.get_device_name())
        pass

    main(__author__)
    plt.show()
