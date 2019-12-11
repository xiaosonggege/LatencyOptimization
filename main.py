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
        client_num=2000,
        x_range=(0, 100),
        y_range=(0, 100),
        V_range=(-6, 6),
        V_local_range=(5, 100),
        V_mec=20,
        T_epsilon=100,
        Q_MEC=5000,
        vector_DMECmap=rng.randint(low=4, high=10, size=2000), #和client_num一致
        B=20,
        P=30,
        h=0.8,
        N0=5
    )
    vector_alpha_init = rng.uniform(low=0, high=1, size=(1, 2000))
    T_TH = 10000
    # print(vector_alpha_init)
    res = lantencymap.solve_problem(vector_alpha=vector_alpha_init, op_function='SLSQP', T_TH=T_TH)
    print('最优时延结果为: %s' % res.fun)
    print('取得最优时延时优化参数向量为:\n', res.x)
    print('迭代次数为: %s' % res.nit)
    print('迭代成功？ %s' % res.success)
if __name__ == '__main__':
    main()


