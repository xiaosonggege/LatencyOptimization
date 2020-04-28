#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: plotting
@time: 2020/4/28 2:50 下午
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot1():
    ddpg = None
    with open(file=r'/Users/songyunlong/Desktop/ddpg200.txt', mode='r') as f:
        while True:
            line_str = f.readline()
            if not line_str:
                break
            line_array = np.array([float(i) for i in line_str.split(' ')])
            ddpg = line_array if ddpg is None else np.vstack((ddpg, line_array))
    with open(file=r'/Users/songyunlong/Desktop/slsqp.txt', mode='r') as f:
        line_str = f.readline()
        slsqp = np.array([float(i) for i in line_str.split(' ')])
    # fig, ax = plt.subplots()
    # x = [i for i in range(1, 101)]
    # ax.plot(x, ddpg[:, -1], c='r', label='ddpg')
    # ax.plot(x, slsqp, c='b', label='slsqp')
    # ax.legend(loc='upper right')
    # ax.set_xlabel('x/time')
    # ax.set_ylabel('y/s')
    # ax.grid(axis='x', linestyle='-.')
    # ax.grid(axis='y', linestyle='-.')
    # fig.show()

    fig, ax = plt.subplots()
    x = [i for i in range(1, 101)]
    ax.plot(x, ddpg[:, 0], c='r', label='reward')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/reward')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

if __name__ == '__main__':
    plot1()
