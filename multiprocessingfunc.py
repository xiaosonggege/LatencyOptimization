#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: multiprocessingfunc
@time: 2020/2/16 11:38 上午
'''
import numpy as np
import multiprocessing
import copy
import re
import os

# with open(file='/Users/songyunlong/Desktop/text.txt', mode='r') as f:
#     listr = f.readlines()
# listr = [lis[:-1] for lis in listr]
#
# print(listr)
# regex = re.compile(pattern='\d+.\d+')
# result = regex.finditer(string=listr[0])
# result1 = list([i.group(0) for i in result])
# print(result1)
# print(np.array([float(i) for i in result1]).sum())
path_vxy = '/Users/songyunlong/Desktop/vxy.txt'
path_Te = '/Users/songyunlong/Desktop/Te.txt'
path_client = '/Users/songyunlong/Desktop/client.txt'
path_B = '/Users/songyunlong/Desktop/vxy.txt'

def fun(x, lock):
    print('begin')
    lock.acquire()
    print(x)
    if not os.path.exists(path_vxy):
        os.mknod(path_vxy)
    with open(file='/Users/songyunlong/Desktop/vxy.txt', mode='a') as f:
        f.write('%s %s' % (x[0], x[-1]) + '\n')
    lock.release()
    print('end')

class datagenerator:
    def __init__(self, func):
        """
        构造函数
        :param func: 单线程执行函数
        """
        self._func = func
        # self._vxy_client_range = None
        # self._T_epsilon = None
        # self._client_num = None
        # self._B = None
        # self._arg_name = None

    def name(self, *arg_info):
        self._arg_name = arg_info[0]
        self.__dict__[self._arg_name] = arg_info[-1]

    def multiprocess(self):
        """
        多线程生成数据
        :return: None
        """
        lock = multiprocessing.Lock()
        ps = []
        for arg in self.__dict__[self._arg_name]:
            p = multiprocessing.Process(target=self._recordingdata, args=(arg, lock))
            p.start()
            ps.append(p)
        [p.join() for p in ps]

    def _recordingdata(self, arg, mutex):
        print(arg)
        if self._arg_name == 'vxy_client_range':
            result = self._func(vxy_client_range=arg)
        elif self._arg_name == 'T_epsilon':
            result = self._func(T_epsilon=arg)
        elif self._arg_name == 'client_num':
            result = self._func(client_num=arg)
        else:
            result = self._func(B=arg)
        mutex.acquire()
        with open(file=path_client, mode='a') as f:
            f.write('%s %s' % (arg, result))
        mutex.release()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=6)
    lock = multiprocessing.Lock()
    l1 = [float(i) for i in range(10)]
    l2 = copy.deepcopy(l1)
    l3 = []
    for i in list(zip(l1, l2)):
        p = multiprocessing.Process(target=fun, args=(i, lock))
        p.start()
        l3.append(p)
    [p.join() for p in l3]