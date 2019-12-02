#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: Environment
@time: 2019/12/2 10:11 下午
@desc:
'''
import numpy as np
class Client:
    '''
    属性：
    V_local: 用户本地执行速率
    V: 用户移动速度矢量
    axis: 用户移动速度矢量
    distance2MECserver: 用户距离边缘服务器距离
    vector_alpha: 用户子任务在本地/MEC端执行的分配比例序列
    vector_D: 用户需要计算的子任务序列
    D_all: 用户需要计算的总任务量
    N: 子任务个数
    move_range: 用户在服务器的服务范围覆盖下的移动范围
    '''
    def __init__(self, V_local, Vx, Vy, axis_x, axis_y, MECserverPostion):
        '''
        用户类构造函数
        :param V_local: 用户本地执行速率
        :param Vx: 用户移动速度x维度矢量值
        :param Vy: 用户移动速度y维度矢量值
        :param axis_x: 用户当前坐标x维度值
        :param axis_y: 用户当前坐标y纬度值
        :param MECserverPostion: 边缘服务器对象
        '''
        self.__V = np.array([Vx, Vy])
        self.__axis = np.array([axis_x, axis_y])
        self.__distance2MECserver = np.sqrt((self.__axis[0]-MECserverPostion()[-2])**2 +
                                            (self.__axis[-1]-MECserverPostion()[-1])**2)
        self.__move_range = 2 * np.sqrt(MECserverPostion()[0]**2 - self.__distance2MECserver**2) / np.sqrt(np.sum(self.__V**2))
        #self.__vector_alpha/self.__vector_D/self.__D_all/self.__N在方法中初始化
    def __str__(self):
        return '用户当前坐标为: (%s, %s), 移动速度为: (%s, %s)' % \
               (self.__axis[0], self.__axis[-1], self.__V[0], self.__V[-1])
    def _calc_tasknum(self, T_epsilon, *otherclient_vector):
        '''
        用户通过时间阈值筛选需要与哪些用户进行时间距离的计算
        :param T_epsilon: 时间阈值
        :param otherclient_vector: 其它用户类向量
        :return: None
        '''
        pass

class MECServer:
    def __init__(self, server_r, V_MEC, Q, *axis):
        '''
        边缘服务器构造函数
        :param server_r: 边缘服务器可以服务的半径范围
        :param V_MEC: 边缘服务器执行任务速率
        :param Q: 边缘服务器存储剩余量
        :param axis: 边缘服务器位置坐标
        '''
        self.__server_r = server_r
        self.__V_MEC = V_MEC
        self.__Q = Q
        self.__axis = np.array([axis])
    def __call__(self):
        '''
        输出服务器的所有属性信息
        :return: np.ndarray, [server_r, V_MEC, Q, axis_x, axis_y]
        '''
        return np.array([self.__server_r, self.__V_MEC, self.__Q, self.__axis[0], self.__axis[-1]])
class LatencyMap:
    pass