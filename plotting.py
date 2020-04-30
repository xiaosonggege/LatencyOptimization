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
import re
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
#插值
spline = lambda x, y: interp1d(x, y, kind='quadratic')
x_spline_new = np.linspace(1, 100, 1000)
def plot1():
    """
    ddpg最好结果和slsqp对比
    :return:
    """
    ddpg = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200.txt', mode='r') as f:
        while True:
            line_str = f.readline()
            if not line_str:
                break
            line_array = np.array([float(i) for i in line_str.split(' ')])
            ddpg = line_array if ddpg is None else np.vstack((ddpg, line_array))
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/slsqp.txt', mode='r') as f:
        line_str = f.readline()
        slsqp = np.array([float(i) for i in line_str.split(' ')])
    fig, ax = plt.subplots()
    x = [i for i in range(1, 101)]
    ax.plot(x_spline_new, spline(x, ddpg[:, -1])(x_spline_new), c='r', label='ddpg')
    ax.plot(x_spline_new, spline(x, slsqp)(x_spline_new), c='b', label='slsqp')
    ax.legend(loc='upper right')
    ax.set_xlabel('x/episode')
    ax.set_ylabel('y/average latency')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

    fig, ax = plt.subplots()
    x = [i for i in range(1, 101)]
    ax.plot(x_spline_new, spline(x, ddpg[:, 0]*100)(x_spline_new), c='r', label='reward')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/reward')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

def plot2(regex = re.compile(pattern='-*\d+\.\d+')):
    """
    lr对ddpg收敛性的影响
    :return:
    """
    ddpg_lr1 = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200.txt', mode='r') as f:
        #lr=1e-3
        while True:
            line_str = f.readline()
            if not line_str:
                break
            line_array = np.array([float(i) for i in line_str.split(' ')])
            ddpg_lr1 = line_array if ddpg_lr1 is None else np.vstack((ddpg_lr1, line_array))

    ddpg_lr2 = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200-1e-2.txt', mode='r') as f:
        #lr=1e-2
        while True:
            line_str = f.readline()
            if not line_str:
                break
            generator = regex.finditer(string=line_str)
            line_array = np.array([float(i.group(0)) for i in generator])
            ddpg_lr2 = line_array if ddpg_lr2 is None else np.vstack((ddpg_lr2, line_array))

    ddpg_lr3 = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200-5e-3.txt', mode='r') as f:
        # lr=5e-3
        while True:
            line_str = f.readline()
            if not line_str:
                break
            generator = regex.finditer(string=line_str)
            line_array = np.array([float(i.group(0)) for i in generator])
            ddpg_lr3 = line_array if ddpg_lr3 is None else np.vstack((ddpg_lr3, line_array))

    #reward
    fig, ax = plt.subplots()
    x = range(1, 101)
    ax.plot(x_spline_new, spline(x, ddpg_lr1[:, 0]*100)(x_spline_new), c='r', label='lr=1e-3')
    ax.plot(x_spline_new, spline(x, ddpg_lr2[:, 0]*100)(x_spline_new), c='g', label='lr=1e-2')
    ax.plot(x_spline_new, spline(x, ddpg_lr3[:, 0]*100)(x_spline_new), c='b', label='lr=5e-3')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/reward')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

    #latency
    fig, ax = plt.subplots()
    x = range(1, 101)
    ax.plot(x_spline_new, spline(x, ddpg_lr1[:, -1])(x_spline_new), c='r', label='lr=1e-3')
    ax.plot(x_spline_new, spline(x, ddpg_lr2[:, -1])(x_spline_new), c='g', label='lr=1e-2')
    ax.plot(x_spline_new, spline(x, ddpg_lr3[:, -1])(x_spline_new), c='b', label='lr=5e-3')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/average latency')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

def plot3(regex = re.compile(pattern='-*\d+\.\d+')):
    """
    batch_size对ddpg收敛性的影响
    :return:
    """
    ddpg_batch1 = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200.txt', mode='r') as f:
        # batch_size=1000
        while True:
            line_str = f.readline()
            if not line_str:
                break
            line_array = np.array([float(i) for i in line_str.split(' ')])
            ddpg_batch1 = line_array if ddpg_batch1 is None else np.vstack((ddpg_batch1, line_array))

    ddpg_batch2 = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200-500.txt', mode='r') as f:
        # batch_size=500
        while True:
            line_str = f.readline()
            if not line_str:
                break
            generator = regex.finditer(string=line_str)
            line_array = np.array([float(i.group(0)) for i in generator])
            ddpg_batch2 = line_array if ddpg_batch2 is None else np.vstack((ddpg_batch2, line_array))

    ddpg_batch3 = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200-700.txt', mode='r') as f:
        # batch_size=700
        while True:
            line_str = f.readline()
            if not line_str:
                break
            generator = regex.finditer(string=line_str)
            line_array = np.array([float(i.group(0)) for i in generator])
            ddpg_batch3 = line_array if ddpg_batch3 is None else np.vstack((ddpg_batch3, line_array))

    # reward
    fig, ax = plt.subplots()
    x = range(1, 101)
    ax.plot(x_spline_new, spline(x, ddpg_batch1[:, 0] * 100)(x_spline_new), c='r', label='batch size=1000')
    ax.plot(x_spline_new, spline(x, ddpg_batch2[:, 0] * 100)(x_spline_new), c='g', label='batch size=500')
    ax.plot(x_spline_new, spline(x, ddpg_batch3[:, 0] * 100)(x_spline_new), c='b', label='batch size=700')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/reward')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

    # latency
    fig, ax = plt.subplots()
    x = range(1, 101)
    ax.plot(x_spline_new, spline(x, ddpg_batch1[:, -1])(x_spline_new), c='r', label='batch size=1000')
    ax.plot(x_spline_new, spline(x, ddpg_batch2[:, -1])(x_spline_new), c='g', label='batch size=500')
    ax.plot(x_spline_new, spline(x, ddpg_batch3[:, -1])(x_spline_new), c='b', label='batch size=700')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/average latency')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()


def plot4(regex = re.compile(pattern='-*\d+\.\d+')):
    """
    reward对ddpg收敛性的影响
    :return:
    """
    ddpg_ut = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200.txt', mode='r') as f:
        # ut
        while True:
            line_str = f.readline()
            if not line_str:
                break
            line_array = np.array([float(i) for i in line_str.split(' ')])
            ddpg_ut = line_array if ddpg_ut is None else np.vstack((ddpg_ut, line_array))

    ddpg_sigmode = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200-sigmoid.txt', mode='r') as f:
        # sigmode
        while True:
            line_str = f.readline()
            if not line_str:
                break
            generator = regex.finditer(string=line_str)
            line_array = np.array([float(i.group(0)) for i in generator])
            ddpg_sigmode = line_array if ddpg_sigmode is None else np.vstack((ddpg_sigmode, line_array))

    ddpg_tanh = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200-tanh.txt', mode='r') as f:
        # tanh
        while True:
            line_str = f.readline()
            if not line_str:
                break
            generator = regex.finditer(string=line_str)
            line_array = np.array([float(i.group(0)) for i in generator])
            ddpg_tanh = line_array if ddpg_tanh is None else np.vstack((ddpg_tanh, line_array))

    # reward
    fig, ax = plt.subplots()
    x = range(1, 101)
    ax.plot(x_spline_new, spline(x, ddpg_ut[:, 0] * 100)(x_spline_new), c='r', label='ε(x)')
    ax.plot(x_spline_new, spline(x, ddpg_sigmode[:, 0] * 100)(x_spline_new), c='g', label='sigmode(x)')
    ax.plot(x_spline_new, spline(x, ddpg_tanh[:, 0] * 100)(x_spline_new), c='b', label='tanh(x)')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/reward')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

    # latency
    fig, ax = plt.subplots()
    x = range(1, 101)
    ax.plot(x_spline_new, spline(x, ddpg_ut[:, -1])(x_spline_new), c='r', label='ε(x)')
    ax.plot(x_spline_new, spline(x, ddpg_sigmode[:, -1])(x_spline_new), c='g', label='sigmode(x)')
    ax.plot(x_spline_new, spline(x, ddpg_tanh[:, -1])(x_spline_new), c='b', label='tanh(x)')
    ax.legend()
    ax.set_xlabel('x/eposide')
    ax.set_ylabel('y/average latency')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()

if __name__ == '__main__':
    # plot1()
    # plot2()
    # plot3()
    plot4()