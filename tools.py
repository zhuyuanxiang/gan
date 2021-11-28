"""
=================================================
@path   : gan -> tools.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2021-11-24 17:54
@Version: v0.1
@License: (C)Copyright 2020-2021, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from datetime import datetime


def func_time(func):
    def wrapper(*args, **kw):
        start_time = datetime.now()  # ----->函数运行前时间
        func(*args, **kw)
        end_time = datetime.now()  # ----->函数运行后时间
        cost_time = end_time - start_time  # ---->运行函数消耗时间
        print("函数%s()消耗时间为%s" % (func.__name__, cost_time))

    return wrapper  # ---->装饰器其实是对闭包的一个应用


def main(name):
    print(f'Hi, {name}', datetime.now())
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
