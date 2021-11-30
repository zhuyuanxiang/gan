"""
=================================================
@path   : gan -> cgan_mnist.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-29 18:14
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

from datasets import generate_random_image
from datasets import generate_random_one_hot
from datasets import generate_random_seed
from datasets import mnist_dataset


class cGANDiscriminator(nn.Module):
    def __init__(self):
        super(cGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(784 + 10, 200),
                nn.LeakyReLU(0.02),
                nn.LayerNorm(200),
                nn.Linear(200, 1),
                nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter = 0
        self.progress = []

    def forward(self, image_tensor, label_tensor):
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)

    def train(self, inputs, label_tensor, targets):
        outputs = self.forward(inputs, label_tensor)
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print("counter = ", self.counter)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


class cGANGenerator(nn.Module):
    def __init__(self):
        super(cGANGenerator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(100 + 10, 200),
                nn.LeakyReLU(0.02),
                nn.LayerNorm(200),
                nn.Linear(200, 784),
                nn.Sigmoid()
        )
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter = 0
        self.progress = []

    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)

    def train(self, D, inputs, label_tensor, targets):
        g_output = self.forward(inputs, label_tensor)
        d_output = D.forward(g_output, label_tensor)
        loss = D.loss_function(d_output, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print("counter = ", self.counter)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.1, marker='.', gird=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


def plot_images(label, G: cGANGenerator):
    label_tensor = torch.zeros((10))
    label_tensor[label] = 1.0
    f, axarr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            axarr[i, j].imshow(G.forward(generate_random_seed(100), label_tensor).detach().numpy().reshape(28, 28),
                               interpolation='none', cmap='Blues')


def main(name):
    print(f'Hi, {name}', datetime.now())

    # test_discriminator()
    test_generator()

    D = cGANDiscriminator()
    G = cGANGenerator()

    epochs = 12

    for epoch in range(epochs):
        print(f"--->第{epoch}次迭代<---")
        for label, image_data_tensor, label_tensor in mnist_dataset:
            D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))
            random_label = generate_random_one_hot(10)
            D.train(G.forward(generate_random_seed(100), random_label).detach(), random_label, torch.FloatTensor([0.0]))
            random_label = generate_random_one_hot(10)
            G.train(D, generate_random_seed(100), random_label, torch.FloatTensor([1.0]))

    D.plot_progress()
    G.plot_progress()

    plot_images(1, G)
    plot_images(3, G)
    plot_images(5, G)
    plot_images(7, G)
    plot_images(9, G)
    pass


def test_generator():
    G = cGANGenerator()
    output = G.forward(generate_random_seed(100), generate_random_one_hot(10))
    img = output.detach().numpy().reshape(28, 28)
    plt.imshow(img, interpolation='none', cmap='Blues')


def test_discriminator():
    D = cGANDiscriminator()
    for label, image_data_tensor, label_tensor in mnist_dataset:
        D.train(image_data_tensor, label_tensor, torch.FloatTensor([1.0]))
        D.train(generate_random_image(784), generate_random_one_hot(10), torch.FloatTensor([0.0]))
    D.plot_progress()

    for i in range(4):
        label, image_data_tensor, label_tensor = mnist_dataset[random.randint(0, 60000)]
        print(D.forward(image_data_tensor, label_tensor).item())
    for i in range(4):
        print(D.forward(generate_random_image(784), generate_random_one_hot(10)).item())


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    plt.show()
