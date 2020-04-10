#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: DynamicRoadNetwork
@time: 2020/4/10 10:30 下午
'''
import numpy as np
from RoadNetwork.Environment import Map

class DynamicEnvironment:
    def __index__(self):
        self._x_map = 1e5
        self._y_map = 1e5
        self._client_num = 3000
        self._MECserver_num = 4
        self._R_client_mean = 1e3 #HZ
        self._R_MEC_mean = 1e5 #Hz  #单个计算任务量均值在1000bit
        self._vxy_client_range = (-60, 60)
        self._T_epsilon = 300 #s
        self._Q_client = 1e2
        self._Q_MEC = 1e3 * self._client_num
        self._server_r = 1 / np.sqrt(2*self._MECserver_num) * self._x_map
        self._r_edge_th = self._server_r * (2 - np.sqrt(2))
        self._B = 6.3e+6
        self._N0 = 1e-10
        self._P = 1e-6
        self._h = 0.95
        self._delta = -0.9
        self.map = Map(
            x_map=self._x_map,
            y_map=self._y_map,
            client_num=self._client_num,
            MECserver_num=self._MECserver_num,
            R_client_mean=self._R_client_mean,
            R_MEC_mean=self._R_MEC_mean,
            vxy_client_range=self._vxy_client_range,
            T_epsilon=self._T_epsilon,
            Q_client=self._Q_client,
            Q_MEC=self._Q_MEC,  # 够承载10000用户所有计算任务的
            server_r=self._server_r,
            r_edge_th=self._r_edge_th,
            B=self._B,
            N0=self._N0,
            P=self._P,
            h=self._h,
            delta=self._delta
        )

    def change_environment(self):
        #将移动速度在当前值基础上增减
        #将本地计算速率在当前值基础上增减
        #将本地任务量存储阈值在当前值基础上增减
        #将云端计算速率在当前值基础上增减
        #将云端任务量存储阈值在当前那值基础上增减
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass



if __name__ == '__main__':
    pass