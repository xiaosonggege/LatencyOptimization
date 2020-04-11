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
    #各个可变值的阈值边界
    Rc_max = 1e3 #Hz
    vxy_max = 60 #km/h
    Qc_max = 1e2 #
    Rm_max = 1e5 #Hz
    Qm_max = 1e3 * 1000 #1000为用户数，实际可以考虑修改

    @staticmethod
    def next_state(loc, scale):
        next_value = np.random.normal(loc=loc, scale=scale)
        if next_value > loc:
            next_value = 2 * loc - next_value
        return next_value

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
        """
        动态改变环境
        :return: v0x, v0y, Rc, Qc, Rm, Qm, Qc_real, Q_m_real, latency
        """
        next_state_func = np.frompyfunc(func=DynamicEnvironment.next_state, nin=2, nout=1)
        #将移动速度在当前值基础上增减, Dx=21
        #历史速度
        clients_v = self.map.clients_v
        clients_v_new = next_state_func(clients_v, 21)
        #除目标client之外其它client的位置更新
        self.map.clients_pos += clients_v_new
        #将本地计算速率在当前值基础上增减, Dx=100
        R_client = self.map.clients_R_client
        R_client_new = next_state_func(R_client, 100)
        #将本地任务量存储阈值在当前值基础上增减, DX=15
        obclient_Q_client = self.map.ob_client.Q_client
        obclient_Q_client_new = next_state_func(obclient_Q_client, 15)
        #将云端计算速率在当前值基础上增减, Dx=6000
        mecservers_R_MEC = self.map.MECservers_R_MEC
        mecservers_R_MEC_new = next_state_func(mecservers_R_MEC, 6000)
        #将云端任务量存储阈值在当前那值基础上增减, Dx=10000
        q_MEC = self.map.Q_MEC
        q_MEC_new = next_state_func(q_MEC, 10000)
        #除目标用户外的其它用户属性更新
        self.map.client_vector = (R_client_new, clients_v_new, clients_v_new[:, 0], clients_v_new[:, -1])
        #更新MEC服务器
        self.map.mecserver_vector = (mecservers_R_MEC_new, q_MEC_new)
        #获取目标client的属性，计算其更新属性
        obclient = self.map.ob_client
        r_client = obclient.R_client
        r_client_new = next_state_func(r_client, 100)
        v = obclient.v
        v_new = next_state_func(v, 21)
        axis = obclient.axis
        x_new, y_new = axis[0] - v_new[0], axis[-1] - v_new[-1]
        #latency
        latency = self.map.solve_problem(R_client=r_client_new, v_x=v_new[0], v_y=v_new[-1],
                                         x_client=x_new, y_client=y_new)
        return obclient.v[0], obclient.v[-1], r_client_new, obclient.Q_res(),\
               mecservers_R_MEC_new, self.map.mecserver_for_obclient.Q_res(), latency


    def __iter__(self):
        return self

    def __next__(self):
        pass



if __name__ == '__main__':
    pass