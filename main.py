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
from Environment import Client
def fun(b, **kwargs):
    '''
    
    '''
    print(kwargs)
if __name__ == '__main__':
    class A:
        def __init__(self, a, b):
            self.__a = a
            self.__b = b
        def __call__(self):
            return self.__a, self.__b
        def __str__(self):
            return 'a:%s, b:%s' % (self.__a, self.__b)

    b = A(1, 2)
    c, d = b()
    print(b)

