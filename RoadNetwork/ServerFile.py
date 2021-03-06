#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: ServerFile
@time: 2020/1/3 4:01 下午
'''
import numpy as np

class Server:
    """
    服务器基类构造函数
    x_server: 服务器位置坐标x分量
    y_server: 服务器位置坐标y分量
    """
    def __init__(self, x_server, y_server):
        """
        服务器基类构造函数
        :param x_server: 服务器位置坐标x分量
        :param y_server: 服务器位置坐标y分量
        """
        self.__x_server = x_server
        self.__y_server = y_server
        self.__client_vector = [] #服务范围内所有client(可能不包括目标client)上传速度和位置信息

    @property
    def axis(self):
        """
        返回服务器位置坐标
        :return: tuple，服务器位置坐标
        """
        return self.__x_server, self.__y_server

    @axis.setter
    def axis(self, xy):
        """
        设置服务器位置坐标
        :param xy: tuple，服务器位置坐标
        :return: None
        """
        self.__x_server, self.__y_server = xy

    @property
    def client_vector(self):
        """
        返回服务范围内client上传的信息
        :return: list，由服务范围内client上传的信息组成的张量
        """
        return self.__client_vector

    @client_vector.setter
    def client_vector(self, clients):
        """
        接收服务范围内client上传的信息
        :param clients: 用户序列
        :return: None
        """
        self.__client_vector = clients

class MECServer(Server):
    """
    MEC服务器类型
    service_r: 服务器的服务范围半径
    R_MEC: MEC服务器cpu计算速率
    Q_MEC: MEC服务器计算任务量阈值
    r_edge_th: 服务范围边缘范围阈值
    """
    rng = np.random.RandomState(0)
    def __init__(self, x_server, y_server, service_r, R_MEC, Q_MEC, r_edge_th):
        """
        MEC服务器构造函数
        :param x_server: 服务器位置坐标x分量
        :param y_server: 服务器位置坐标y分量
        :param service_r: 服务器服务范围半径
        :param R_MEC: MEC服务器cpu计算速率
        :param Q_MEC: MEC服务器计算任务量阈值
        :param r_edge_th: 服务范围边缘范围阈值
        """
        super().__init__(x_server=x_server, y_server=y_server)
        self.__service_r = service_r
        self.__R_MEC = R_MEC
        self.__Q_MEC = Q_MEC
        self.__r_edge_th = r_edge_th
        self.__Q_used = MECServer.rng.uniform(low=0, high=1) #根据实际情况修改

    @property
    def r_edge_th(self):
        return self.__r_edge_th

    @property
    def service_r(self):
        """
        返回服务器服务范围半径
        :return: 服务器服务范围半径
        """
        return self.__service_r

    @service_r.setter
    def service_r(self, r):
        """
        设置服务器服务范围半径
        :param r: 服务器服务范围半径
        :return: None
        """
        self.__service_r = r

    @property
    def R_MEC(self):
        """
        返回MEC服务器cpu计算速率
        :return: MEC服务器cpu计算速率
        """
        return self.__R_MEC

    def Q_res(self):
        """
        返回MEC服务器当前剩余存储容量
        :return: MEC服务器当前剩余存储容量
        """
        return self.__Q_MEC - self.__Q_used
    @property
    def Q_MEC(self):
        return self.__Q_MEC

    def client_pos_to_MECserver(self, client):
        """
        判断client处在MEC服务器的位置区域
        :param client: 目标client
        :return: int，1表示client处于MEC服务器中非边缘处，0表示client处于MEC服务器边缘处，-1表示client不处于MEC服务器的服务范围内
        """
        dis = np.sqrt(np.sum((np.array(client.axis) - np.array(self.axis)) ** 2))
        if dis <= self.__r_edge_th:
            return 1
        elif dis > self.__r_edge_th and dis <= self.__service_r:
            return 0
        return -1

    def dis_to_centerserver(self, x_server, y_server):
        """
        计算MECserver到Centerserver的欧式距离
        :param x_server: Centerserver位置坐标x分量
        :param y_server: Centerserver位置坐标y分量
        :return: MECserver到Centerserver的欧式距离
        """
        return np.sqrt(np.sum((np.array(self.axis) - np.array([x_server, y_server]) ** 2)))

    def MEC_calc_time(self, D_MEC):
        """
        计算MEC端计算任务所需时间
        :param D_MEC: MEC端计算任务量
        :return: MEC端计算任务所需时间
        """
        return D_MEC / self.__R_MEC



class CenterServer(Server):
    """
    中心服务器
    x_server: 服务器位置坐标x分量
    y_server: 服务器位置坐标y分量
    T_epsilon: 时延阈值
    """
    def __init__(self, x_server, y_server, T_epsilon):
        """
        中心服务器构造函数
        :param x_server: 服务器位置坐标x分量
        :param y_server: 服务器位置坐标y分量
        :param T_epsilon: 时延阈值
        """
        super().__init__(x_server=x_server, y_server=y_server)
        self.__T_epsilon = T_epsilon

    def filter_client_vector(self, obclient):
        """
        筛选出需要与目标client之间进行临近检测的其它client
        :return: list，由需要与目标client之间进行临近检测的其他client组成的列表
        """
        # print(type(self.client_vector[0]))
        v = np.array([client.v for client in self.client_vector])
        v_max_mod = np.max(np.sqrt(np.sum(v ** 2, axis=1)))
        v_obclient_mod = np.sqrt(np.sum(np.array(obclient.v) ** 2))
        #计算中心服务器的服务范围半径
        self.__server_r = (v_max_mod + v_obclient_mod) * self.__T_epsilon
        #从地图中所有client中筛选出需要进行临近检测的client组成列表返回
        position_clients = np.array([client.axis for client in self.client_vector])
        clients_satisfied_index = np.where(np.sqrt(np.sum((position_clients - np.array(obclient.axis)) ** 2, axis=1))
                                     < self.__server_r, 1, 0)
        # clients_satisfied = []
        clients_satisfied_index = [index for index, val in enumerate(clients_satisfied_index) if val == 1]
        clients_satisfied = [self.client_vector[index] for index in clients_satisfied_index]
        return clients_satisfied


if __name__ == '__main__':
    s1 = Server(1, 2)
    s2 = MECServer(1, 2, 3, 4, 5, 6)
