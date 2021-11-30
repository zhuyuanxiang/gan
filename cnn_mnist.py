"""
=================================================
@path   : gan -> cnn_mnist.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-29 14:39
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
import matplotlib.pyplot as plt
from datetime import datetime

import pandas
import torch
import torch.nn as nn

from datasets import mnist_dataset
from datasets import mnist_test_dataset


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=(5, 5), stride=(2, 2)),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(10),
                nn.Conv2d(10, 10, kernel_size=(3, 3), stride=(2, 2)),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(10),
                nn.Flatten(0, -1),
                nn.Linear(250, 10),
                nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters())

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print("counter =", self.counter)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))


def main(name):
    print(f'Hi, {name}', datetime.now())
    C = Classifier()

    epochs = 3

    for i in range(epochs):
        print("training epoch", i + 1, "of", epochs)
        for label, image_data_tensor, target_tensor in mnist_dataset:
            C.train(image_data_tensor.view(1, 1, 28, 28), target_tensor)

    C.plot_progress()
    record = 19
    mnist_test_dataset.plot_image(record)
    image_data = mnist_test_dataset[record][1]
    output = C.forward(image_data.view(1, 1, 28, 28))
    pandas.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0, 1))

    score, items = 0, 0

    for label, image_data_tensor, target_tensor in mnist_test_dataset:
        answer = C.forward(image_data_tensor.view(1, 1, 28, 28)).detach().numpy()
        if answer.argmax() == label:
            score += 1
        items += 1
    print(score, items, score / items)
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
    plt.show()
