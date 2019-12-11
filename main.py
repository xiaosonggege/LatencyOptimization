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
        client_num=5000,
        x_range=(0, 1e3),
        y_range=(0, 1e3),
        V_range=(0, 10),
        V_local_range=(10000, 20000),
        V_mec=200000,
        T_epsilon=10, #对比
        Q_MEC=1e+4,
        vector_DMECmap=rng.randint(low=1e+2, high=2e+2, size=5000), #和client_num一致
        B=6.3e+6,
        P=1e-6,
        h=0.95,
        N0=1e-10
    )
    vector_alpha_init = rng.uniform(low=0, high=1, size=(1, 5000))
    T_TH = 10000
    # print(vector_alpha_init)
    res = lantencymap.solve_problem(vector_alpha=vector_alpha_init, op_function='SLSQP', T_TH=T_TH)
    print('最优时延结果为: %s' % res.fun)
    print('取得最优时延时优化参数向量为:\n', res.x)
    print('迭代次数为: %s' % res.nit)
    print('迭代成功？ %s' % res.success)
if __name__ == '__main__':
    main()


