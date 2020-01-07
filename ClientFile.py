#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: ClientFile
@time: 2020/1/3 4:01 下午
'''
import numpy as np

class Client:
    """
    用户基类
    :parameter:
    R_client: 用户本地cpu计算速率
    v_x: 用户移动速度x分量
    v_y: 用户移动速度y分量
    x_client: 用户位置坐标x分量
    y_client: 用户位置坐标y分量
    """
    def __init__(self, R_client, v_x, v_y, x_client, y_client):
        """
        用户类型构造函数
        :param R_client: 用户本地cpu计算速率
        :param v_x: 用户移动速度x分量
        :param v_y: 用户移动速度y分量
        :param x_client: 用户位置坐标x分量
        :param y_client: 用户位置坐标y分量
        """
        self.__R_client = R_client
        self.__v_x = v_x
        self.__v_y = v_y
        self.__x_client = x_client
        self.__y_client = y_client

    @property
    def v(self):
        """
        返回用户速度信息
        :return: tuple，用户速度矢量
        """
        return self.__v_x, self.__v_y

    @v.setter
    def v(self, *v):
        """
        用户速度设置
        :param v: tuple，用户速度矢量
        :return: None
        """
        self.__v_x, self.__v_y = v

    @property
    def axis(self):
        """
        返回用户位置信息
        :return: tuple，用户位置矢量
        """
        return self.__x_client, self.__y_client

    @axis.setter
    def axis(self, *xy):
        """
        用户位置设置
        :param xy: tuple，用户位置矢量
        :return: None
        """
        self.__x_client, self.__y_client = xy

    @property
    def R_client(self):
        """
        返回用户本地cpu计算速率
        :return: 用户本地cpu计算速率
        """
        return self.__R_client

class ObjectClient(Client):
    """
    目标用户类型
    :parameter:
    R_client: 用户本地cpu计算速率
    v_x: 用户移动速度x分量
    v_y: 用户移动速度y分量
    x_client: 用户位置坐标x分量
    y_client: 用户位置坐标y分量
    R_client: 用户本地cpu计算速率
    v_x: 用户移动速度x分量
    v_y: 用户移动速度y分量
    x_client: 用户位置坐标x分量
    y_client: 用户位置坐标y分量
    D_vector: ndarray，待处理的任务序列
    x_server: 边缘服务器位置x分量
    y_server: 边缘服务器位置y分量
    alpha_vector: ndarray，子任务序列的权值分配
    Q_client: 用户计算任务量阈值
    """
    rng = np.random.RandomState(0)
    def __init__(self, R_client, v_x, v_y, x_client, y_client, D_vector, x_server, y_server, alpha_vector, Q_client):
        """
        目标用户类型构造函数
        :param R_client: 用户本地cpu计算速率
        :param v_x: 用户移动速度x分量
        :param v_y: 用户移动速度y分量
        :param x_client: 用户位置坐标x分量
        :param y_client: 用户位置坐标y分量
        :param D_vector: ndarray，待处理的任务序列，需根据其它临近用户位置速度信息生成
        :param x_server: 边缘服务器位置x分量
        :param y_server: 边缘服务器位置y分量
        :param alpha_vector: ndarray，子任务序列的权值分配
        :param Q_client: 用户计算任务量阈值
        """
        super().__init__(R_client=R_client, v_x=v_x, v_y=v_y, x_client=x_client, y_client=y_client)
        self.__D_vector = D_vector
        self.__x_server = x_server
        self.__y_server = y_server
        self.__D_vector_length = self.__D_vector.size
        self.__alpha_vector = alpha_vector
        self.__Q_client = Q_client
        self.__Q_used = ObjectClient.rng.uniform(low=0, high=1) #根据实际情况修改

    def Q_res(self):
        """
        返回目标client当前剩余存储容量
        :return: 目标client当前剩余存储容量
        """
        return self.__Q_client - self.__Q_used

    def dis_to_MECserver(self, x_server, y_server, service_r):
        """
        计算目标client到为其服务的MECserver的欧氏距离
        :param x_server: MECserver位置坐标x分量
        :param y_server: MECserver位置坐标y分量
        :param service_r: MECserver服务范围半径
        :return: 目标client到为其服务的MECserver的欧式距离
        """
        pass #改

    def task_distributing(self):
        """
        按权值向量分配本地任务量和需要卸载到MEC服务器端的任务量
        :return: 需要卸载的任务量
        """
        return np.sum(self.__D_vector * (1 - self.__alpha_vector))

    def local_calc_time(self):
        """
        计算本地计算任务所需时间
        :return: 本地计算任务所需时间
        """
        self.__D_local = np.sum(self.__D_vector * self.__alpha_vector)
        return self.__D_local / self.R_client


if __name__ == '__main__':
    c1 = Client(1, 2, 3, 4, 5)
    print(c1.v)
    c2 = ObjectClient(1, 2, 3, 4, 5, np.array([6, 6]), 7, 8, 9, 10)
    print(c2.v)