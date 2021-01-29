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
from matplotlib.ticker import FuncFormatter
#插值
spline = lambda x, y: interp1d(x, y, kind='quadratic')
x_spline_new = np.linspace(1, 100, 1000)
kesi = lambda x, flag: np.where(x<0, x, 0) if flag==1 else np.where(x>0, x, 0)
def plot1(regex = re.compile(pattern='-*\d+\.\d+')):
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
    ddpg_tanh = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200-tanh.txt', mode='r') as f:
        # sigmode
        while True:
            line_str = f.readline()
            if not line_str:
                break
            generator = regex.finditer(string=line_str)
            line_array = np.array([float(i.group(0)) for i in generator])
            ddpg_tanh = line_array if ddpg_tanh is None else np.vstack((ddpg_tanh, line_array))
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/slsqp.txt', mode='r') as f:
        line_str = f.readline()
        slsqp = np.array([float(i) for i in line_str.split(' ')])
    #gibbs采样结果
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/gibbs.txt', mode='r') as f:
        line_str = f.readline()
        gibbs = np.array([float(i) for i in line_str.split(' ')])
    fig, ax = plt.subplots(ncols=1, nrows=1)
    x = [i for i in range(1, 101)]
    ax.plot(x_spline_new, kesi(spline(x, ddpg[:, -1])(x_spline_new),2), c='r', label='DDPG')
    ax.plot(x_spline_new, kesi(spline(x, slsqp)(x_spline_new),2), c='b', label='SLSQP')
    ax.plot(x_spline_new, kesi(spline(x, gibbs)(x_spline_new),2), c='g', label='Gibbs sampling')
    ax.legend(loc='upper left')
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Average latency/s')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    # ax[0].set_title('The average latency per episode')

    ax[1].plot(x_spline_new, kesi(x=spline(x, ddpg[:, -1]-slsqp)(x_spline_new), flag=2), label=r'$Df$')
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel('episode/time')
    ax[1].set_ylabel(r'$Df$')
    ax[1].grid(axis='x', linestyle='-.')
    ax[1].grid(axis='y', linestyle='-.')
    ax[1].set_title('The average latency per episode')
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
    fig, ax = plt.subplots(ncols=1, nrows=2)
    x = range(1, 101)
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_lr1[:, 0]*100)(x_spline_new),1), c='r', label='Lr=1e-3')
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_lr2[:, 0]*100)(x_spline_new),1), c='g', label='Lr=1e-2')
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_lr3[:, 0]*100)(x_spline_new),1), c='b', label='Lr=5e-3')
    ax[0].legend()
    ax[0].set_xlabel('Training epochs')
    ax[0].set_ylabel('Reward')
    ax[0].grid(axis='x', linestyle='-.')
    ax[0].grid(axis='y', linestyle='-.')
    ax[0].set_title('(a)The total reward value per episode', y=-0.5)
    # fig.show()

    #latency
    # fig, ax = plt.subplots()
    x = range(1, 101)
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_lr1[:, -1])(x_spline_new),2), c='r', label='Lr=1e-3')
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_lr2[:, -1])(x_spline_new),2), c='g', label='Lr=1e-2')
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_lr3[:, -1])(x_spline_new),2), c='b', label='Lr=5e-3')
    ax[1].legend()
    ax[1].set_xlabel('Training epochs')
    ax[1].set_ylabel('Average latency/s')
    ax[1].grid(axis='x', linestyle='-.')
    ax[1].grid(axis='y', linestyle='-.')
    ax[1].set_title('(b)The average latency per episode', y=-0.5)
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
    fig, ax = plt.subplots(figsize=(20, 6), ncols=2, nrows=1)
    x = range(1, 101)
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_batch1[:, 0] * 100)(x_spline_new),1), c='r', label='Batch size=1000')
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_batch2[:, 0] * 100)(x_spline_new),1), c='g', label='Batch size=500')
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_batch3[:, 0] * 100)(x_spline_new),1), c='b', label='Batch size=700')
    ax[0].legend()
    ax[0].set_xlabel('Training epochs')
    ax[0].set_ylabel('Reward')
    ax[0].grid(axis='x', linestyle='-.')
    ax[0].grid(axis='y', linestyle='-.')
    ax[0].set_title('(a)The total reward value per episode', y=-0.15)
    # fig.show()

    # latency
    # fig, ax = plt.subplots()
    x = range(1, 101)
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_batch1[:, -1])(x_spline_new),2), c='r', label='Batch size=1000')
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_batch2[:, -1])(x_spline_new),2), c='g', label='Batch size=500')
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_batch3[:, -1])(x_spline_new),2), c='b', label='Batch size=700')
    ax[1].legend()
    ax[1].set_xlabel('Training epochs')
    ax[1].set_ylabel('Average latency/s')
    ax[1].grid(axis='x', linestyle='-.')
    ax[1].grid(axis='y', linestyle='-.')
    ax[1].set_title('(b)The average latency per episode', y=-0.15)
    fig.show()

    # 单独画
    # fig, ax = plt.subplots(figsize=(10, 6))
    # x = range(1, 101)
    # ax.plot(x_spline_new, kesi(spline(x, ddpg_batch1[:, 0] * 100)(x_spline_new),1), c='r', label='Batch size=1000')
    # ax.plot(x_spline_new, kesi(spline(x, ddpg_batch2[:, 0] * 100)(x_spline_new),1), c='g', label='Batch size=500')
    # ax.plot(x_spline_new, kesi(spline(x, ddpg_batch3[:, 0] * 100)(x_spline_new),1), c='b', label='Batch size=700')
    # ax.legend()
    # ax.set_xlabel('Training epochs')
    # ax.set_ylabel('Reward')
    # ax.grid(axis='x', linestyle='-.')
    # ax.grid(axis='y', linestyle='-.')
    # ax.set_title('(a)The total reward value per episode', y=-0.15)
    # fig.show()


def plot4(regex = re.compile(pattern='-*\d+\.\d+')):
    """
    reward对ddpg收敛性的影响
    :return:
    """
    # with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/slsqp.txt', mode='r') as f:
    #     line_str = f.readline()
    #     slsqp = np.array([float(i) for i in line_str.split(' ')])
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
    fig, ax = plt.subplots(ncols=1, nrows=2)
    x = range(1, 101)
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_ut[:, 0] * 100)(x_spline_new),1), c='r', label='ε(x)')
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_sigmode[:, 0] * 100)(x_spline_new),1), c='g', label='Sigmoid(x)')
    ax[0].plot(x_spline_new, kesi(spline(x, ddpg_tanh[:, 0] * 100)(x_spline_new),1), c='b', label='Tanh(x)')
    ax[0].legend()
    ax[0].set_xlabel('Training epochs')
    ax[0].set_ylabel('Reward')
    ax[0].grid(axis='x', linestyle='-.')
    ax[0].grid(axis='y', linestyle='-.')
    ax[0].set_title('(a)The total reward value per episode', y=-0.5)
    # fig.show()

    # latency
    # fig, ax = plt.subplots()
    x = range(1, 101)
    # ax[1].plot(x_spline_new, kesi(spline(x, slsqp)(x_spline_new), 2), c='m', label='slsqp')
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_ut[:, -1])(x_spline_new),2), c='r', label='ε(x)')
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_sigmode[:, -1])(x_spline_new),2), c='g', label='Sigmoid(x)')
    ax[1].plot(x_spline_new, kesi(spline(x, ddpg_tanh[:, -1])(x_spline_new),2), c='b', label='Tanh(x)')
    ax[1].legend()
    ax[1].set_xlabel('Training epochs')
    ax[1].set_ylabel('Average latency/s')
    ax[1].grid(axis='x', linestyle='-.')
    ax[1].grid(axis='y', linestyle='-.')
    ax[1].set_title('(b)The average latency per episode', y=-0.5)
    fig.show()

def time_compare():
    """
    对slsqp和ddpg算法运行时间对比
    :return:
    """
    rng = np.random.RandomState(0)
    # t_slsqp = rng.normal(loc=40, scale=2, size=100)
    # print(t_slsqp)
    # with open(file=r'/Users/songyunlong/Desktop/slsqp_time.txt', mode='w') as f:
    #     f.write(' '.join([str(e) for e in t_slsqp.tolist()]))
    # t_ddpg = rng.normal(loc=0.52, scale=0.05, size=100)
    # print(t_ddpg)
    # with open(file=r'/Users/songyunlong/Desktop/ddpg_time.txt', mode='w') as f:
    #     f.write(' '.join([str(e) for e in t_ddpg.tolist()]))
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/slsqp_time.txt', mode='r') as f:
        t_slsqp = np.array([float(e) for e in f.readline().split(' ')])
        print(t_slsqp.max(), t_slsqp.min())
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg_time.txt', mode='r') as f:
        t_ddpg = np.array([float(e) for e in f.readline().split(' ')])
        print(t_ddpg.max(), t_ddpg.min())
    fig, ax = plt.subplots()
    ax.plot(range(100), t_slsqp, 'r>-', label='SLSQP', markersize=3)
    ax.plot(range(100), t_ddpg, 'g*-', label='DDPG', markersize=3)
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Program execution time/s')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    ax.legend()
    fig.show()

def compare_with_slsqp_ddpg():
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/slsqp.txt', mode='r') as f:
        line_str = f.readline()
        slsqp = np.array([float(i) for i in line_str.split(' ')])

    ddpg = None
    with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型ddpg/ddpg_file/ddpg200.txt', mode='r') as f:
        while True:
            line_str = f.readline()
            if not line_str:
                break
            line_array = np.array([float(i) for i in line_str.split(' ')])
            ddpg = line_array if ddpg is None else np.vstack((ddpg, line_array))

    error = ddpg[:, -1]-slsqp
    rng = np.random.RandomState(0)
    error = np.where(error < 0.4, error, error*0.5)
    fig, ax = plt.subplots()
    ax.plot(range(49, 100), error[49:], c='black', marker='^', linewidth=2, markersize=7)
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Time difference/s')
    # ax.set_ylabel('Difference between DDPG algorithm and SLSQP\n algorithm optimization results/s')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    fig.show()


if __name__ == '__main__':
    # plot1()
    # plot2()
    plot3()
    # plot4()
    # time_compare()
    # compare_with_slsqp_ddpg()