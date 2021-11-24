# -*- encoding: utf-8 -*-
import torch
import torch

"""
=================================================
@path   : gan -> pytorch_basics.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-24 15:34
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from datetime import datetime
import torch


def main(name):
    print("===> python variables <===")
    python_variables(name)

    print("===> python tensors <===")
    pytorch_tensors()

    print("===> pytorch computation graph <===")
    pytorch_computation_graph()
    pass


def pytorch_computation_graph():
    x = torch.tensor(3.5, requires_grad=True)
    y = x * x
    z = 2 * y + 3
    z.backward()
    print("x.grad=", x.grad)
    # print("y.grad=", y.grad)    # 非根结点，不能输出其梯度
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = 2 * a + 3 * b
    y = 5 * a * a + 3 * b * b * b
    z = 2 * x + 3 * y
    z.backward()
    print("a.grad=", a.grad)
    print("b.grad=", b.grad)


def pytorch_tensors():
    x = torch.tensor(3.5)
    y = x + 3
    z = torch.tensor(3.5, requires_grad=True)

    print("pytorch tensor x=", x)
    print("x.requires_grad=", x.requires_grad)
    print("y=x+3=", y)
    print("pytorch tensor z=", z)
    print("z.requires_grad=", z.requires_grad)

    t1 = (x - 1) * (x - 2) * (x - 3)
    print("t1=", t1)
    # print("---> after backward <---")
    # t1.backward()
    # print("pytorch tensor after backward x=", x)

    t2 = (z - 1) * (z - 2) * (z - 3)
    print("t2=", t2)
    t2.backward()
    print("---> after backward <---")
    print("pytorch tensor z=", z)
    print("pytorch tensor z.grad=", z.grad)
    print("t2=", t2)


def python_variables(name):
    print(f'Hi, {name}', datetime.now())
    x = 3.5
    y = x * x + 2
    print("x=", x)
    print("y=", y)


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
