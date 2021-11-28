# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
import torch

"""
=================================================
@path   : gan -> gan_simple.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-25 11:28
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from datetime import datetime
import torch
import torch.nn as nn

import pandas
import matplotlib.pyplot as plt
import random
import numpy
from tools import func_time
import pprint


def generate_real():
    real_data = torch.FloatTensor([
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2),
            random.uniform(0.8, 1.0),
            random.uniform(0.0, 0.2)
    ])
    return real_data


def generate_random(size):
    random_data = torch.rand(size)
    return random_data


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(4, 3),
                nn.Sigmoid(),
                nn.Linear(3, 1),
                nn.Sigmoid()
        )

        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []
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
            print("counter = ", self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass

    pass


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(1, 3),
                nn.Sigmoid(),
                nn.Linear(3, 4),
                nn.Sigmoid()
        )
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D: Discriminator, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)
        loss = D.loss_function(d_output, targets)
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        self.optimiser.zero_grad()
        # print("before backward d_output=", D.forward(g_output))
        loss.backward()
        # print("after backward d_output=", D.forward(g_output))
        self.optimiser.step()

    def plot_gress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))


def main(name):
    print(f'Hi, {name}', datetime.now())

    # test_discriminator()
    # test_generator()

    train_GAN()
    pass


@func_time
def train_GAN():
    D = Discriminator()
    G = Generator()
    image_list = []
    for i in range(16000):
        D.train(generate_real(), torch.FloatTensor([1.0]))
        D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
        G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
        if i % 1000 == 0:
            image_list.append(G.forward(torch.FloatTensor([0.5])).detach().numpy())
    D.plot_progress()
    G.plot_gress()
    # print(G.forward(torch.FloatTensor([0.5])))
    plt.figure(figsize=(16, 8))
    plt.imshow(numpy.array(image_list).T, interpolation='none', cmap='Blues')
    pprint.pprint(image_list)


def test_generator():
    G = Generator()
    print(G.forward(torch.FloatTensor([0.5])))


def test_discriminator():
    D = Discriminator()
    for i in range(16000):
        D.train(generate_real(), torch.FloatTensor([1.0]))
        D.train(generate_random(4), torch.FloatTensor([0.0]))
        pass
    D.plot_progress()
    real_data = generate_real()
    print("real_data=", real_data)
    print(D.forward(real_data).item())
    random_data = generate_random(4)
    print("random_data=", random_data)
    print(D.forward(random_data).item())


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    plt.show()
