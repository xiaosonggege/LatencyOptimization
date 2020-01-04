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

    """
    def __init__(self, x_server, y_server, service_r):
        """
        服务器基类构造函数
        :param x_server: 服务器位置坐标x分量
        :param y_server: 服务器位置坐标y分量
        :param service_r: 服务器的服务范围半径
        """
        self.__x_server = x_server
        self.__y_server = y_server
        self.__service_r = service_r

    @property
    def axis(self):
        """
        返回服务器位置坐标
        :return: tuple，服务器位置坐标
        """
        return self.__x_server, self.__y_server

    @axis.setter
    def axis(self, *xy):
        self.__x_server, self.__y_server = xy


class MECServer(Server):
    pass

class CenterServer(Server):
    pass

if __name__ == '__main__':
    pass