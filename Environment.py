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
from scipy.optimize import minimize

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
    filter_D: 确定用户需要计算和哪些其它用户之间的计算任务
    N: 子任务个数
    movetime_range: 用户在服务器的服务范围覆盖下的移动时间范围
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
        self.__V = np.array([Vx, Vy])  # 需要可见
        self.__axis = np.array([axis_x, axis_y])  # 需要可见
        self.__distance2MECserver = np.sqrt((self.__axis[0] - MECserverPostion.getaxis[0]) ** 2 +
                                            (self.__axis[-1] - MECserverPostion.getaxis[-1]) ** 2)
        self.__movetime_range = 2 * np.sqrt(MECserverPostion.getr ** 2 - self.__distance2MECserver ** 2) / np.sqrt(
            np.sum(self.__V ** 2))
        self.__vector_D = None  # 需要可见
        self.__D_all = None
        self.__N = None  # 需要可见
        self.__filter_D = None

    def __call__(self):
        """
        用户信息
        """
        print('V_local: %s\nV: %s\naxis: %s\ndistance2MECserver: %s\nmovetime_range: %s\nvector_D: %s\nD_all: %s\nN: %s\nfilter_D: %s' %
              (self.__V_local, self.__V, self.__axis, self.__distance2MECserver, self.__movetime_range, self.__vector_D, self.__D_all,
               self.__N, self.__filter_D))
    # 可见操作模块
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

    def getvector_D(self):
        ''''''
        return self.__vector_D

    def setvector_D(self, vector_D):
        """"""
        self.__D = vector_D

    vector_D = property(getvector_D, setvector_D)

    # def getN(self):
    #     """"""
    #     return self.__N
    #
    # def setN(self, N):
    #     """"""
    #     self.__N = N
    #
    # N = property(getN, setN)
    def getmovetime_range(self):
        """"""
        return self.__movetime_range
    movetime_range = property(getmovetime_range)
    #

    def calc_tasknum(self, T_epsilon, otherclient_vector):
        """
        用户通过时间阈值筛选需要与哪些用户进行时间距离的计算
        :param T_epsilon: 时间阈值
        :param otherclient_vector: 其它用户类向量
        :return: None
        """
        # 初始化任务向量D_allmapclient_i = (x, y, Vx, Vy)
        D_allMECmapclient = [[oc.axis[0], oc.axis[-1], oc.V[0], oc.V[-1]] for oc in otherclient_vector]
        D_allMECmapclient = np.array(D_allMECmapclient)
        # print(D_allMECmapclient.shape)
        all_axis, all_v = np.array_split(D_allMECmapclient, 2, axis=1)
        # print(all_axis.shape, all_v.shape)
        S = np.sqrt(np.sum((all_axis - self.axis[np.newaxis, :]) ** 2, axis=1))
        # print(S.shape)
        V_m_add = np.sqrt(np.sum(all_v ** 2, axis=1)) + np.sqrt(np.sum(self.__V ** 2))
        dis_t = S / V_m_add
        self.__filter_D = np.where(dis_t - T_epsilon < 0, 1, 0)
        self.__filter_D = np.argwhere(self.__filter_D == 1)
        self.__filter_D = self.__filter_D.ravel()
        self.__N = self.__filter_D.size

    def calc_else_attri(self, vector_D_allMECmap):
        """
        计算vector_D、D_all属性
        :param vector_D_allMECmap: MEC服务器服务范围内的全部用户
        return: None
        """
        self.__vector_D = vector_D_allMECmap[self.__filter_D]
        self.__D_all = np.sum(self.__vector_D)

    def calc(self, vector_alpha):
        """
        计算t_local,此方法只用于当前考虑的用户
        :param vector_alpha: 子任务量比例分配向量
        :return: t_local
        """
        D_local = self.__D_all - np.sum(vector_alpha * self.__vector_D)
        return D_local / self.__V_local


class MECServer:
    """
    属性:
    server_r: 边缘服务器可以服务的半径范围
    V_MEC: 边缘服务器执行任务速率
    Q: 边缘服务器存储剩余量
    axis: 边缘服务器位置坐标
    """
    @property
    def task_vector(self):
        return self.__task_vector
    @task_vector.setter
    def task_vector(self, task_vector):
        """"""
        self.__task_vector = task_vector
    @property
    def getaxis(self):
        return self.__axis
    @property
    def getr(self):
        return self.__server_r
    def __init__(self, server_r, V_MEC, Q, B, P, h, N0, *axis):
        '''
        边缘服务器构造函数
        :param server_r: 边缘服务器可以服务的半径范围
        :param V_MEC: 边缘服务器执行任务速率
        :param Q: 边缘服务器存储剩余量
        :param B: 传输信道带宽
        :param P: 发射功率
        :param h: 信道增益
        :param N0: 高斯白噪声功率谱密度
        :param axis: 边缘服务器位置坐标
        '''
        self.__server_r = server_r
        self.__V_MEC = V_MEC
        self.__Q = Q
        self.__axis = np.array([axis])
        self.__task_vector = None
        self.__B = B
        self.__P = P
        self.__h = h
        self.__N0 = N0

    def __call__(self):
        """
        输出服务器的所有属性信息
        """
        print('server_r: %s\nV_MEC: %s\nQ: %s\naxis: %s\ntask_vector: %s\nB: %s\nP: %s\nh: %s\nN0: %s' %
              (self.__server_r, self.__V_MEC, self.__Q, self.__axis, self.__task_vector, self.__B, self.__P, self.__h, self.__N0))

    def calc_D_MEC(self, alpha_vector):
        """
        计算D_MEC
        :param alpha_vector: 子任务量比例分配向量
        """
        return np.sum(alpha_vector * self.__task_vector)

    def calc_t_MEC(self, alpha_vector):
        """
        计算t_MEC,其中需要计算上行传输速率，忽略下行传输速率
        :param alpha_vector: 子任务量比例分配向量
        :return: t_MEC
        """
        D_MEC = self.calc_D_MEC(alpha_vector)
        t_up = D_MEC / (self.__B * np.log2(1 + (self.__P * self.__h ** 2) / self.__N0))
        t_work = D_MEC / self.__V_MEC
        return t_up + t_work


class LatencyMap:
    def __init__(self, client_num, x_range, y_range, V_range, V_local_range, V_mec,
                 T_epsilon, Q_MEC, vector_DMECmap, B, P, h, N0):
        """
        :param client_num: 单个MEC服务器可以服务的用户数量
        :param x_range: 单个服务器服务范围的x方向范围
        :param y_range: 单个服务器服务范围的y方向范围
        :param V_range: 用户移动速度矩阵，shape=(1, 2)
        :param V_local_range: 单个用户本地CPU执行速率范围
        :param V_mec: MEC服务器CPU执行速率
        :param T_epsilon: 时间距离阈值
        :param Q_MEC: MEC服务器存储容量最大值
        :param vector_DMECmap: MEC服务器服务范围内的全部用户与该用户之间需要计算的临近性任务量向量
        :param B: 传输信道带宽
        :param P: 发射功率
        :param h: 信道增益
        :param N0: 高斯白噪声功率谱密度
        """
        self.__client_num = client_num
        self.__x_range = x_range
        self.__y_range = y_range
        self.__V_range = V_range
        self.__V_local_range = V_local_range
        self.__V_mec = V_mec
        self.__T_epsilon = T_epsilon
        self.__Q_MEC = Q_MEC
        self.__vector_DMECmap = vector_DMECmap
        self.__B = B
        self.__P = P
        self.__h = h
        self.__N0 = N0
        self.__this_client = None
        self.__clients = None
        self.__MEC = None
        self.__T = None  # 总时延

    def _build_model(self):
        """
        建立静态场景
        :return: None
        """
        rng = np.random.RandomState(0)
        param_tensor = lambda param_range, param_size: rng.uniform(low=param_range[0], high=param_range[-1], size=param_size)
        x_client = param_tensor(self.__x_range, self.__client_num)
        # print(x_client)
        y_client = param_tensor(self.__y_range, self.__client_num)
        # print(y_client)
        V_client = param_tensor(self.__V_range, param_size=(self.__client_num, 2)) #用户移动速度
        # print(V_client)
        V_local_client = param_tensor(self.__V_local_range, param_size=self.__client_num)
        # print(V_local_client)
        #服务器服务半径为坐标范围对角线半径
        server_r = 0.5 * np.sqrt((self.__x_range[-1]-self.__x_range[0])**2 + (self.__y_range[-1]-self.__y_range[0])**2)
        server_x = (self.__x_range[-1] - self.__x_range[0]) / 2
        server_y = (self.__y_range[-1] - self.__y_range[0]) / 2
        # print(server_r, server_x, server_y)
        self.__MEC = MECServer(server_r, self.__V_mec, self.__Q_MEC, self.__B,
                               self.__P, self.__N0, self.__h, server_x, server_y)
        # print(self.__MEC())
        #生成client_num个用户
        self.__clients = [Client(V_local=v_local, Vx=vx, Vy=vy, axis_x=x, axis_y=y, MECserverPostion=self.__MEC)
                   for v_local, vx, vy, x, y in zip(V_local_client, V_client[:, 0], V_client[:, -1], x_client, y_client)]
        #本用户
        self.__this_client = self.__clients[-1]
        #筛选该用户需要与哪些其它用户执行临近性任务,此时该用户的N、filter_D属性会被计算出来
        self.__this_client.calc_tasknum(T_epsilon=self.__T_epsilon, otherclient_vector=self.__clients)
        #此时该用户的vector_D、D_all属性会被计算出来
        self.__this_client.calc_else_attri(vector_D_allMECmap=self.__vector_DMECmap)
        # self.__this_client()
        self.__MEC.task_vector = self.__this_client.vector_D

    def _calc_T(self, vector_alpha):
        """
        计算总时延
        :param vector_alpha: 子任务量比例分配向量
        :return: T
        """
        t_total = self.__this_client.calc(vector_alpha=vector_alpha)
        # print(self.__this_client.N)
        t_MEC = self.__MEC.calc_t_MEC(alpha_vector=vector_alpha)
        T = np.max(np.array(t_total, t_MEC))
        return T

    def solve_problem(self, vector_alpha, op_function, T_TH):
        """
        :param op_function:
        'Nelder-Mead':单纯行法
        'Powell'
        'CG'
        'BFGS'
        'Newton-CG'
        'L-BFGS-B'
        'TNC'
        'COBYLA'
        'SLSQP'
        'trust-constr'
        'dogleg'
        'trust-ncg'
        'trust-exact'
        'trust-krylov'
        'custom - a callable object'
        :return None
        :param vector_alpha: 子任务量比例分配向量
        :param T_TH: 对总时延的约束
        return None
        """
        self._build_model()
        fun = lambda alpha : self._calc_T(alpha)
        #约束项函数
        # 约束条件 分为eq 和ineq
        # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
        cons = [{'type': 'ineq', 'fun': lambda alphas: self.__Q_MEC - self.__MEC.calc_D_MEC(alphas)},
                {'type': 'ineq',
                 'fun': lambda alphas: self.__this_client.movetime_range - self.__MEC.calc_t_MEC(alphas)},
                {'type': 'ineq', 'fun': lambda alphas: T_TH - fun(alphas)},
                {'type': 'ineq', 'fun': lambda alphas: alphas.T},
                {'type': 'ineq', 'fun': lambda alphas: - alphas.T + 1}]

        # print(len(cons))
        res = minimize(fun, vector_alpha, method=op_function, constraints=cons)
        return res



