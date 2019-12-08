#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: main
@time: 2019/11/29 1:47 下午
@desc:
'''
import numpy as np
import matplotlib as plt
# import tensorflow as tf
# import torch
import sys
from Environment import Client, MECServer, LatencyMap
def main():
    """"""
    rng = np.random.RandomState(0)
    lantencymap = LatencyMap(
        client_num=10,
        x_range=(0, 100),
        y_range=(0, 100),
        V_range=(-6, 6),
        V_local_range=(5, 10),
        V_mec=20,
        T_epsilon=30,
        Q_MEC=50,
        vector_DMECmap=rng.randint(low=4, high=20, size=10), #和client_num一致
        B=20,
        P=30,
        h=0.8,
        N0=5
    )
    vector_alpha_init = rng.uniform(low=0, high=1, size=(1, 10))
    # print(vector_alpha_init)
    res = lantencymap.solve_problem(vector_alpha=vector_alpha_init, op_function='SLSQP')
    print(res.fun)
    print(res.x)
    print(res.success)
if __name__ == '__main__':
    main()


