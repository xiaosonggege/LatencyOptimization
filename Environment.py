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
        self.__V_local = V_local
        self.__V = np.array([Vx, Vy]) #需要可见
        self.__axis = np.array([axis_x, axis_y]) #需要可见
        self.__distance2MECserver = np.sqrt((self.__axis[0]-MECserverPostion()[-2])**2 +
                                            (self.__axis[-1]-MECserverPostion()[-1])**2)
        self.__move_range = 2 * np.sqrt(MECserverPostion()[0]**2 - self.__distance2MECserver**2) / np.sqrt(np.sum(self.__V**2))
        self.__vector_alpha = None #需要可见
        self.__vector_D = None #需要可见
        self.__D_all = None
        self.__N = None #需要可见
    #可见操作模块
    def getV(self):
        ''''''
        return self.__V
    def setV(self, V):
        ''''''
        self.__V = V
    V = property(getV, setV)
    def getaxis(self):
        ''''''
        return self.__axis
    def setaxis(self, axis):
        ''''''
        self.__axis = axis
    axis = property(getaxis, setaxis)
    def getvector_alpha(self):
        ''''''
        return self.__vector_alpha
    def setvector_alpha(self, vector_alpha):
        ''''''
        self.__vector_alpha = vector_alpha
    vector_alpha = property(getvector_alpha, setvector_alpha)
    def getvector_D(self):
        ''''''
        return self.__vector_D
    def setvector_D(self, vector_D):
        ''''''
        self.__D = vector_D
    vector_D = property(getvector_D, setvector_D)
    def getN(self):
        ''''''
        return self.__N
    def setN(self, N):
        ''''''
        self.__N = N
    N = property(getN, setN)
    #
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
        #初始化任务向量D_allmapclient_i = (x, y, Vx, Vy)
        D_allmapclient = [[oc.axis[0], oc.axis[-1], oc.V[0], oc.V[-1]] for oc in otherclient_vector]
        D_allmapclient = np.array(D_allmapclient)

    def _calc(self):
        '''
        计算t_local,次方法只用于当前考虑的用户
        :return: t_local
        '''
        D_local = self.__D_all - np.sum(self.__vector_alpha * self.__vector_D)
        return D_local / self.__V_local
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
    def _calc(self, alpha_vector, task_vector, B, P, h, N0):
        '''
        计算t_MEC,其中需要计算上行传输速率，忽略下行传输速率
        :param alpha_vector: 子任务量比例分配向量
        :param task_vctor: 子任务量向量
        :param B: 传输信道带宽
        :param P: 发射功率
        :param h: 信道增益
        :param N0: 高斯白噪声功率谱密度
        :return: t_MEC
        '''
        D_MEC = alpha_vector * task_vector
        t_up = D_MEC / (B * np.log2(1 + (P * h **2) / N0))
        t_work = D_MEC / self.__V_MEC
        return t_up + t_work

class LatencyMap:
    pass