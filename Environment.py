#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: Environment
@time: 2020/1/3 4:00 下午
'''
import numpy as np
from ClientFile import Client, ObjectClient
from ServerFile import Server, MECServer, CenterServer
class Map:
    """
    场景地图
    :parameter
    x_map: 地图长度
    y_map: 地图宽度
    client_num: 地图中client数量
    MECserver_num: 地图中MECserver数量，默认MECserver均匀分布在地图中
    R_client_mean: client中cpu计算速率均值
    R_MEC_mean: MECserver中cpu计算速率均值
    vxy_client_range: client移动速度分量范围
    T_epsilon: 时间阈值
    Q_client: client计算任务量阈值
    Q_MEC: MECserver计算任务量阈值
    r_edge_th: 服务范围边缘范围阈值
    B: 无线信道带宽
    N0: 高斯白噪声单边功率谱密度
    P: 发射功率
    h: 信道增益
    delta: 发射功率随距离信源距离的衰减系数
    """
    rng = np.random.RandomState(0)
    param_tensor = lambda param_range, param_size: Map.rng.uniform(low=param_range[0], high=param_range[-1],
                                                               size=param_size)
    param_tensor_gaussian = lambda mean, var, param_size: Map.rng.normal(loc=mean, scale=var, size=param_size)

    def __init__(self, x_map, y_map, client_num, MECserver_num, R_client_mean, R_MEC_mean,
                 vxy_client_range, T_epsilon, Q_client, Q_MEC, server_r, r_edge_th, B, N0, P, h, delta):
        """
        场景地图构造函数
        :param x_map: 地图长度
        :param y_map: 地图宽度
        :param client_num: 地图中client数量
        :param MECserver_num: 地图中MECserver数量，默认MECserver均匀分布在地图中，数量为平方数
        :param R_client_mean: float，client中cpu计算速率均值
        :param R_MEC_mean: float，MECserver中cpu计算速率均值
        :param vxy_client_range: tuple，client移动速度分量范围
        :param T_epsilon: 时间阈值
        :param Q_client: client计算任务量阈值
        :param Q_MEC: MECserver计算任务量阈值
        :param server_r: 服务器的服务范围半径
        :param r_edge_th: 服务范围边缘范围阈值
        :param B: 无线信道带宽
        :param N0: 高斯白噪声单边功率谱密度
        :param P: 发射功率
        :param h: 信道增益
        :param delta: 发射功率随距离信源距离的衰减系数
        """
        self.__x_map = x_map
        self.__y_map = y_map
        self.__client_num = client_num
        self.__MECserver_num = MECserver_num
        self.__R_client_mean = R_client_mean
        self.__R_MEC_mean = R_MEC_mean
        self.__vxy_client_range = vxy_client_range
        self.__T_epsilon = T_epsilon
        self.__Q_client = Q_client
        self.__Q_MEC = Q_MEC
        self.__server_r = server_r
        self.__r_edge_th = r_edge_th
        self.__B = B
        self.__N0 = N0
        self.__P = P
        self.__h = h
        self.__delta = delta
        #用户速度矩阵
        clients_v = Map.param_tensor(param_range=self.__vxy_client_range, param_size=(self.__client_num-1, 2))

        clients_posx = Map.param_tensor(param_range=(0, self.__x_map), param_size=(self.__client_num-1, 1))
        clients_posy = Map.param_tensor(param_range=(0, self.__y_map), param_size=(self.__client_num-1, 1))
        #用户位置矩阵
        self.__clients_pos = np.hstack((clients_posx, clients_posy))

        #用户cpu计算速率向量
        clients_R_client = Map.param_tensor_gaussian(mean=self.__R_client_mean, var=1, param_size=self.__client_num-1)
        # 子任务序列的权值分配
        self.__alpha_vector = Map.param_tensor(param_range=(0, 1), param_size=[1, self.__client_num-1])

        #client序列
        self.__client_vector = [Client(R_client=R_client, v_x=v_x, v_y=v_y, x_client=x_client, y_client=y_client)
                                for R_client, v_x, v_y, x_client, y_client in
                                zip(
                                    clients_R_client,
                                    clients_v[:, 0],
                                    clients_v[:, -1],
                                    clients_posx,
                                    clients_posy
                                )]

        #MECserver的cpu计算速率向量
        MECservers_R_MEC = Map.param_tensor_gaussian(mean = self.__R_MEC_mean, var=1, param_size=self.__MECserver_num)

        MECservers_posx = np.linspace(0, self.__x_map, 2 + int(np.sqrt(self.__MECserver_num)))[1:-1]
        MECservers_posy = np.linspace(0, self.__y_map, 2 + int(np.sqrt(self.__MECserver_num)))[1:-1]
        #MECserver的位置坐标
        self.__MECservers_pos = np.array([(x, y) for x in MECservers_posx for y in MECservers_posy])

        #MECserver序列
        self.__MECserver_vector = [MECServer(x_server=x_server, y_server=y_server, service_r=service_r, R_MEC=R_MEC,
                                      Q_MEC=Q_MEC, r_edge_th=r_edge_th)
                            for x_server, y_server, service_r, R_MEC, Q_MEC, r_edge_th in
                            zip(
                                self.__MECservers_pos[:, 0],
                                self.__MECservers_pos[:, -1],
                                np.ones(shape=self.__MECserver_num) * self.__server_r,
                                MECservers_R_MEC,
                                np.ones(shape=self.__MECserver_num) * self.__Q_MEC,
                                np.ones(shape=self.__MECserver_num) * self.__r_edge_th
                            )]

    def obclient_producing(self, R_client, v_x, v_y, x_client, y_client):
        """
        目标client生成
        :param R_client: 目标用户本地cpu计算速率
        :param v_x: 目标用户移动速度x分量
        :param v_y: 目标用户移动速度y分量
        :param x_client: 目标用户位置坐标x分量
        :param y_client: 目标用户位置坐标y分量
        :return: None
        """
        #根据目标client的位置选择MECserver
        MECserver_for_obclient = None
        #找出与obclient距离最小的MECserver(序列)
        distance_of_obclient_and_MECservers = np.sum(self.__MECservers_pos - np.array([x_client, y_client]), axis=1)
        min_distance_of_obc_MEC = np.min(distance_of_obclient_and_MECservers)
        min_distance_of_obc_MEC_index = np.argwhere(distance_of_obclient_and_MECservers==min_distance_of_obc_MEC)
        min_distance_of_obc_MEC_index = min_distance_of_obc_MEC_index.ravel()
        MECservers_for_obclient = self.__MECserver_vector[min_distance_of_obc_MEC_index]

        #如果存在多个与obclient距离最小的MECserver，则根据MEC服务器当前时间服务器剩余存储容量
        if len(MECservers_for_obclient) > 1:
            MECservers_Q_res = np.array([MECserver.Q_res() for MECserver in MECservers_for_obclient])
            min_MECservers_Q_res_index = np.argwhere(MECservers_Q_res == np.min(MECservers_Q_res)).ravel()
            MECservers_for_obclient = MECservers_for_obclient[min_MECservers_Q_res_index]
        else:
            MECserver_for_obclient = MECservers_for_obclient[0]

        #如果MECserver仍不唯一，则服务范围内client个数评判目标client选择哪个MEC服务器为其服务
        if len(MECservers_for_obclient) > 1:
            pass #改
        else:
            MECserver_for_obclient = MECservers_for_obclient[0]
        #如果MECserver仍不唯一，则随机选取一个
        if len(MECservers_for_obclient) > 1:
            MECserver_for_obclient = Map.rng.choice(a=MECservers_for_obclient, size=1, replace=False)
        # obclient
        self.__obclient = ObjectClient(
            R_client=R_client,
            v_x=v_x,
            v_y=v_y,
            x_client=x_client,
            y_client=y_client,
            Q_client=self.__Q_client,
            alpha_vector=self.__alpha_vector

        )

if __name__ == '__main__':
    pass