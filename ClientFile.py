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
        self.__v_x = v[0]
        self.__v_y = v[-1]

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
        self.__x_client = xy[0]
        self.__y_client = xy[-1]

class ObjectClient(Client):
    """

    """
    def __init__(self, R_client, v_x, v_y, x_client, y_client, D_vector, x_server, y_server, alpha_vector):
        """
        目标用户类型构造函数
        :param R_client: 用户本地cpu计算速率
        :param v_x: 用户移动速度x分量
        :param v_y: 用户移动速度y分量
        :param x_client: 用户位置坐标x分量
        :param y_client: 用户位置坐标y分量
        :param D_vector: ndarray，待处理的任务序列
        :param x_server: 边缘服务器位置x分量
        :param y_server: 边缘服务器位置y分量
        :param alpha_vector: ndarray，子任务序列的权值分配
        """
        super().__init__(R_client, v_x, v_y, x_client, y_client)
        self.__D_vector = D_vector
        self.__x_server = x_server
        self.__y_server = y_server
        self.__D_vector_length = self.__D_vector.size
        self.__alpha_vector = alpha_vector


if __name__ == '__main__':
    