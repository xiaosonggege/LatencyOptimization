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
        Q_MEC=600,
        vector_DMECmap=rng.randint(low=4, high=20, size=10), #和client_num一致
        B=20,
        P=30,
        h=0.8,
        N0=5
    )
    res = lantencymap.solve_problem(vector_alpha=rng.randn(1, 10), op_function='SLSQP')
    print(res.fun)
    print(res.x)
if __name__ == '__main__':
    main()
    # class A:
    #     def __init__(self):
    #         self.__a = None
    #         self.__b = None
    #     def geta(self):
    #         return self.__a
    #     def seta(self, a):
    #         self.__a = a
    #     a = property(geta, seta)
    #     def getb(self):
    #         return self.__b
    #     def setb(self, b):
    #         self.__b = b
    #     b = property(getb, setb)
    #     def __call__(self):
    #         print(self.__a, self.__b)
    #
    #     def fun(self):
    #         print(self.__a + self.__b)
    #
    # a = A()
    # a.a = 2
    # a.b = 3
    # a()
    # a.fun()

