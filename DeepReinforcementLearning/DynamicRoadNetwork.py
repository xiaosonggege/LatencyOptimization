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

    def _client_status_update(self, clients_v:np.ndarray, clients_pos:np.ndarray):
        """"""
        #临时存储更新后的位置，如果出现越界现象，则后续需要更新
        # print(type(clients_v), clients_v.dtype, type(clients_pos), clients_pos.dtype)
        clients_pos_new_temp = clients_pos + clients_v
        # print('woshi', clients_pos_new_temp.shape)
        need_to_change_vx = ~((clients_pos_new_temp[:, 0] >= 0) & (clients_pos_new_temp[:, 0] <= self._x_map))
        # print(need_to_change_vx.__class__)
        need_to_change_vy = ~((clients_pos_new_temp[:, 1] >= 0) & (clients_pos_new_temp[:, 1] <= self._y_map))
        # print(need_to_change_vy.shape)
        if True in need_to_change_vx:
            # print('x坐标需要修改')
            #将对应x方向出边界的移动用户速度反向，并重新计算x方向位置
            clients_v[need_to_change_vx, 0] = -clients_v[need_to_change_vx, 0]
            # print(clients_v)
        if True in need_to_change_vy:
            # print('y坐标需要修改')
            #将对应y方向出边界的移动用户速度反向，并重新计算y方向位置
            clients_v[need_to_change_vy, -1] = -clients_v[need_to_change_vy, -1]
        # print(type(clients_v), clients_v.dtype, type(clients_pos), clients_pos.dtype)
        clients_pos += clients_v
        return clients_v, clients_pos

    def __init__(self):
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
        :return:
        v0(0): obclient移动速度矢量
        pos0(1): obclient位置坐标
        Rc(2): obclient本地计算速率
        Rm(3): MEC端任务计算速率
        """
        next_state_func = np.frompyfunc(DynamicEnvironment.next_state, 2, 1)
        #将移动速度在当前值基础上增减, Dx=21
        #历史速度
        clients_v = self.map.clients_v
        # print(clients_v.dtype)
        clients_v_new = np.asarray(next_state_func(clients_v, 21), 'float64')
        # print(clients_v_new.shape)
        # print(clients_v_new.dtype)
        #除目标client之外其它client的位置更新
        clients_v_new, clients_pos_new = self._client_status_update(clients_v=clients_v_new, clients_pos=self.map.clients_pos)
        #将本地计算速率在当前值基础上增减, Dx=100
        R_client = self.map.clients_R_client
        R_client_new = np.asarray(next_state_func(R_client, 100), 'float64')
        #将云端计算速率在当前值基础上增减, Dx=6000
        mecservers_R_MEC = self.map.MECservers_R_MEC
        mecservers_R_MEC_new = np.asarray(next_state_func(mecservers_R_MEC, 6000), 'float64')
        #将云端任务量存储阈值在当前那值基础上增减, Dx=10000
        q_MEC = self.map.Q_MEC
        q_MEC_new = np.asarray(next_state_func(q_MEC, 10000), 'float64')
        #除目标用户外的其它用户属性更新
        self.map.client_vector = (R_client_new, clients_v_new, clients_pos_new[:, 0], clients_pos_new[:, -1])
        # print('client_vector is finished')
        #更新MEC服务器
        self.map.mecserver_vector = (mecservers_R_MEC_new, q_MEC_new)
        # print('mecserver_vector is finished')
        #将本地任务量存储阈值在当前值基础上增减, DX=15
        obclient_Q_client = self.map.ob_client.Q_client
        obclient_Q_client_new = np.asarray(next_state_func(obclient_Q_client, 15), 'float64')
        #获取目标client的属性，计算其更新属性
        obclient = self.map.ob_client
        r_client = obclient.R_client
        r_client_new = np.asarray(next_state_func(r_client, 100), 'float64')
        v = obclient.v
        v_new = np.asarray(next_state_func(v, 21), 'float64')
        axis = obclient.axis
        obclient_v_new, obclient_pos_new = self._client_status_update(clients_v=np.array(v_new)[np.newaxis, :],
                                                                      clients_pos=np.array(axis)[np.newaxis, :])
        #latency
        self.map.ob_client.Q_client = obclient_Q_client_new
        return r_client_new, obclient_v_new, obclient_pos_new, mecservers_R_MEC_new, obclient.v[0], obclient.v[-1], \
               obclient.Q_res(), self.map.mecserver_for_obclient.Q_res(), obclient.D_vector

    # def get_state(self):
    #     """
    #
    #     :return:
    #     """
    #     obclient = self.map.ob_client
    #     return obclient.v[0], obclient.v[-1], obclient.Q_res(), self.map.mecserver_for_obclient.Q_res(), \
    #            obclient.D_vector

    def get_alpha(self):
        """

        :return:
        """
        return self.map.ob_client.alpha_vector

    def get_latency(self, r_client_new, obclient_v_new, obclient_pos_new):
        """

        :return:
        """
        client_Q_constraint, mec_Q_constraint, latency = self.map.solve_problem(R_client=r_client_new,
                                                                            v_x=obclient_v_new[0],
                                                                            v_y=obclient_v_new[-1],
                                                                            x_client=obclient_pos_new[0],
                                                                            y_client=obclient_pos_new[-1],
                                                                            op_function='latency')
        return client_Q_constraint, mec_Q_constraint, latency

    def __enter__(self):
        rng = np.random.RandomState(0)
        r_client_new = rng.normal(loc=1e3, scale=1, size=1) #目标用户初始计算速度均值
        obclient_v_new = (10, 10)
        # obclient_pos_new = (97779.54569559613, 88473.93449231013)
        obclient_pos_new = (25000., 25000.)
        self.client_Q_constraint, self.mec_Q_constraint, self.latency = self.get_latency(r_client_new=r_client_new,
                                                                                        obclient_v_new=obclient_v_new,
                                                                                        obclient_pos_new=obclient_pos_new)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

if __name__ == '__main__':
    with DynamicEnvironment() as d:
        for i in range(100):
            r_client_new, obclient_v_new, obclient_pos_new, mecservers_R_MEC_new, *outputs1 = d.change_environment()

            # outputs2 = d.get_state()
            outputs3 = d.get_alpha()
            print(outputs3.shape, outputs1[-1].__len__(), '\n')
            #由于obclient_v_new和obclient_pos_new的维度是二维，所以需flatten
            output4 = d.get_latency(r_client_new=r_client_new, obclient_v_new=obclient_v_new.ravel(),
                                    obclient_pos_new=obclient_pos_new.ravel())
