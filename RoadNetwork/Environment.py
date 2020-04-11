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
import sympy
from scipy.optimize import minimize
import scipy.integrate as si
from RoadNetwork.ClientFile import Client, ObjectClient
from RoadNetwork.ServerFile import Server, MECServer, CenterServer

#对为目标client服务的边缘Server描述符
class AttributePropertyMEC:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance, owner):
        attrs = instance.__dict__[self._name]
        pos, r, r_TH = attrs.axis, attrs.service_r, attrs.r_edge_th
        return pos, r, r_TH

#对所有边缘Server的描述符
class AttributePropertyMEC_series:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance, owner)->tuple:
        """"""
        attrs = instance.__dict__[self._name] #attrs为list结构
        r, r_TH = attrs[0].service_r, attrs[0].r_edge_th #服务半径和服务范围边界区域下限半径
        servers_posx = [mecpos.axis[0] for mecpos in attrs]
        servers_posy = [mecpos.axis[-1] for mecpos in attrs]
        return servers_posx, servers_posy, r, r_TH

#目标client描述符
class AttributePropertyOb:
    def __init__(self, name):
        self._name = name
    def __get__(self, instance, owner):
        return instance.__dict__[self._name].axis
class ClientVectorProperty:
    def __init__(self, name):
        self._name = name
    def __get__(self, instance, owner):
        return instance.__dict__[self._name]
    def __set__(self, instance, value:tuple):
        """
        更新移动用户向量
        :param instance: object of Map
        :param value: tuple，Rc, vx, vy, x, y
        :return: None
        """
        pass

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
    filter_list = [] #记录空间内各client的服务MECserver标志
    @staticmethod
    def clientsForMECserver(client_vector, MECserver):
        """
        确定服务器服务范围内的用户
        :param client_vector: 地图中的所有用户
        :param MECserver: 服务器对象
        :return: None
        """
        # print(client_vector[0].axis)
        clients_pos = np.array([client.axis for client in client_vector])
        # print(clients_pos.shape)
        # a = (clients_pos - np.array(MECserver.axis)) ** 2
        # print(a.shape)
        dis_between_clients_and_MECserver = np.sqrt(np.sum((clients_pos - np.array(MECserver.axis)) ** 2, axis=1))
        # print(dis_between_clients_and_MECserver.shape)
        dis_between_clients_and_MECserver_index = np.argwhere(dis_between_clients_and_MECserver < MECserver.service_r).ravel().tolist()
        #由于每个client只需要找到一个MECserver为其进行服务，所以需要进行判断每个符合条件的client是否已有为其服务的服务器
        dis_between_clients_and_MECserver_index = [i for i in dis_between_clients_and_MECserver_index if Map.filter_list[i] == 0]
        #记录新筛选出的client
        for index in dis_between_clients_and_MECserver_index:
            Map.filter_list[index] = 1

        # print(len(dis_between_clients_and_MECserver_index))
        # print(len(client_vector))
        MECserver.client_vector = [client_vector[index] for index in dis_between_clients_and_MECserver_index]
        # print(len(MECserver.client_vector))

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
        self.__CenterMECserver = CenterServer(x_server=x_map * 2, y_server=y_map * 2, T_epsilon=self.__T_epsilon)
        #用户速度矩阵
        clients_v = Map.param_tensor(param_range=self.__vxy_client_range, param_size=(self.__client_num-1, 2))

        clients_posx = Map.param_tensor(param_range=(0, self.__x_map), param_size=self.__client_num-1)
        clients_posy = Map.param_tensor(param_range=(0, self.__y_map), param_size=self.__client_num-1)
        #用户位置矩阵
        self.__clients_pos = np.hstack((clients_posx[:, np.newaxis], clients_posy[:, np.newaxis])) ##打印

        #用户cpu计算速率向量
        clients_R_client = Map.param_tensor_gaussian(mean=self.__R_client_mean, var=1, param_size=self.__client_num-1)
        # 子任务序列的权值分配初始化为0向量
        # self.__alpha_vector = np.zeros(shape=[1, self.__client_num-1])

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
        # print(self.__client_vector[0].axis)

        #MECserver的cpu计算速率向量
        MECservers_R_MEC = Map.param_tensor_gaussian(mean = self.__R_MEC_mean, var=1, param_size=self.__MECserver_num)
        #边缘服务器间等距且与边界等距，此分法服务半径大，各边缘服务器间交叉区域大
        # MECservers_posx = np.linspace(0, self.__x_map, 2 + int(np.sqrt(self.__MECserver_num)))[1:-1]
        # MECservers_posy = np.linspace(0, self.__y_map, 2 + int(np.sqrt(self.__MECserver_num)))[1:-1]
        #边缘服务器间等距且与边界不等距，此分法服务半径相对小，各边缘服务器间交叉区域小
        EdgePoint_calc = lambda para1, para2: \
            1/(2*np.sqrt(self.__MECserver_num))*para1 + (1-1/(2*np.sqrt(self.__MECserver_num)))*para2
        MECservers_posx = np.linspace(EdgePoint_calc(0, self.__x_map), EdgePoint_calc(self.__x_map, 0), int(np.sqrt(self.__MECserver_num)))
        MECservers_posy = np.linspace(EdgePoint_calc(0, self.__y_map), EdgePoint_calc(self.__y_map, 0), int(np.sqrt(self.__MECserver_num)))
        #MECserver的位置坐标
        self.__MECservers_pos = np.array([(x, y) for x in MECservers_posx for y in MECservers_posy])
        Map.filter_list = [0 for _ in range(client_num)] #初始化记录列表为全0
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
        for MECserver in self.__MECserver_vector:
            Map.clientsForMECserver(client_vector=self.__client_vector, MECserver=MECserver)
            # print(type(MECserver.client_vector), len(MECserver.client_vector))

    @property
    def clients_pos(self):
        return self.__clients_pos

    def point_of_intersection_calculating(self):
        """
        计算目标client移动速度方向直线与MECserver服务范围边界圆的两个交点
        :return: ndarray，交点坐标
        """
        x_1, y_1 = self.__obclient.axis
        vx, vy = self.__obclient.v
        x_2, y_2 = self.__MECserver_for_obclient.axis
        r = self.__MECserver_for_obclient.service_r
        # print(x_1, y_1, vx, vy, x_2, y_2, r)
        x, y = sympy.symbols('x y')
        points = sympy.solve([(x - x_2) ** 2 + (y - y_2) ** 2 - r ** 2, (x - x_1) / vx - (y - y_1) / vy], [x, y])
        # print(points[0][0].evalf(), points[0][-1].evalf(), points[-1][0].evalf(), points[-1][-1].evalf())
        result = np.array([(points[0][0].evalf(), points[0][-1].evalf()), (points[-1][0].evalf(), points[-1][-1].evalf())])
        # print(result.shape)
        return result

    def _obclient_and_MECserver_for_obclient_producing(self, R_client, v_x, v_y, x_client, y_client):
        """
        目标client生成以及为目标client服务的MECserver生成
        :param R_client: 目标用户本地cpu计算速率
        :param v_x: 目标用户移动速度x分量
        :param v_y: 目标用户移动速度y分量
        :param x_client: 目标用户位置坐标x分量
        :param y_client: 目标用户位置坐标y分量
        :return: None
        """
        #根据目标client的位置选择MECserver
        self.__MECserver_for_obclient = self.__MECserver_vector[0] #初始化
        #找出与obclient距离最小的MECserver(序列)
        distance_of_obclient_and_MECservers = np.sqrt(np.sum((self.__MECservers_pos - np.array([x_client, y_client])) ** 2, axis=1))
        min_distance_of_obc_MEC = np.min(distance_of_obclient_and_MECservers)
        min_distance_of_obc_MEC_index = np.argwhere(distance_of_obclient_and_MECservers==min_distance_of_obc_MEC)
        min_distance_of_obc_MEC_index = min_distance_of_obc_MEC_index.ravel().tolist()
        MECservers_for_obclient = [self.__MECserver_vector[index] for index in min_distance_of_obc_MEC_index]

        #如果存在多个与obclient距离最小的MECserver，则根据MEC服务器当前时间服务器剩余存储容量
        if len(MECservers_for_obclient) > 1:
            MECservers_Q_res = np.array([MECserver.Q_res() for MECserver in MECservers_for_obclient])
            min_MECservers_Q_res_index = np.argwhere(MECservers_Q_res == np.min(MECservers_Q_res)).ravel().tolist()
            MECservers_for_obclient = [MECservers_for_obclient[index] for index in min_MECservers_Q_res_index]
        else:
            self.__MECserver_for_obclient = MECservers_for_obclient[0]

        #如果MECserver仍不唯一，则服务范围内client个数评判目标client选择哪个MEC服务器为其服务
        if len(MECservers_for_obclient) > 1:
            for MECserver in MECservers_for_obclient:
                Map.clientsForMECserver(client_vector=self.__client_vector, MECserver=MECserver)
            clients_num = np.array([len(MECserver.client_vector) for MECserver in MECservers_for_obclient])
            min_clients_num_index = np.argwhere(clients_num == np.min(clients_num)).ravel().tolist()
            MECservers_for_obclient = [MECservers_for_obclient[index] for index in min_clients_num_index]
        else:
            self.__MECserver_for_obclient = MECservers_for_obclient[0]
        #如果MECserver仍不唯一，则随机选取一个
        if len(MECservers_for_obclient) > 1:
            MECserver_for_obclient_index = Map.rng.choice(a=np.arange(len(MECservers_for_obclient)), size=1, replace=False)[0]
            self.__MECserver_for_obclient = MECservers_for_obclient[MECserver_for_obclient_index] #打印位置和半径以及边界范围信息

        # obclient
        self.__obclient = ObjectClient(
            R_client=R_client,
            v_x=v_x,
            v_y=v_y,
            x_client=x_client,
            y_client=y_client,
            Q_client=self.__Q_client,
            D_vector=np.zeros(shape=len(self.__MECserver_for_obclient.client_vector)),
            x_server=self.__MECserver_for_obclient.axis[0],
            y_server=self.__MECserver_for_obclient.axis[1]
        )

        axis_a, axis_b = np.split(self.point_of_intersection_calculating(), indices_or_sections=2, axis=0)
        axis_a, axis_b = axis_a[0], axis_b[0]
        t1 = (axis_b[0] - self.__obclient.axis[0]) / self.__obclient.v[0]
        t2 = (axis_b[0] - self.__obclient.axis[0]) / self.__obclient.v[0]
        self.__t_stay = t1 if t1 else t2

    #MECserver_for_obclient描述符
    MECserver_for_obclient = AttributePropertyMEC('_Map__MECserver_for_obclient')
    Obclient = AttributePropertyOb('_Map__obclient')
    MECserver_vector = AttributePropertyMEC_series('_Map__MECserver_vector')
    def transmitting_R(self, is_client=1):
        """
        计算无线信道的传输速率均值
        :param is_client: bool, 指示当前是否是对client和MECserver之间进行计算
        :return: 无线信道传输时延
        """
        t = sympy.symbols('t')
        def f(t):
            """"""
            # if is_client:
            #     d = self.__obclient.dis_to_MECserver(point_of_intersection=self.point_of_intersection_calculating(), t=t)
            # else:
            #     d = self.__MECserver_for_obclient.dis_to_centerserver(
            #         x_server=self.__CenterMECserver.axis[0],
            #         y_server=self.__CenterMECserver.axis[-1]
            #     )
            d = self.__obclient.dis_to_MECserver(point_of_intersection=self.point_of_intersection_calculating(), t=t)
            R_transmit = self.__B * np.log2(1 + self.__P *
                                                np.power(d, -self.__delta) * (self.__h ** 2) / (self.__N0 * self.__B))

            return R_transmit
        # result = sympy.integrate(f, 0, self.__t_stay) / self.__t_stay
        result = si.quad(f, 0, self.__t_stay)[0] / self.__t_stay
        return result

    def time_transmitting_and_MEC_calculating(self, alphas):
        """
        计算任务卸载时间和MECserver计算时间
        :param alphas: 子任务权值分配向量
        :return: 计算任务卸载时间和MECserver计算时间总和
        """
        # 目标client按权值分配需要在本地执行和需要卸载的计算任务
        task_MEC_all = self.__obclient.task_distributing(alphas=alphas)
        # 卸载任务时间
        time_transmitting_calculating = np.sum(self.__obclient.D_vector * (1 - alphas)) / self.transmitting_R()

        # MECserver计算卸载任务所需时间
        time_MEC_calculating = self.__MECserver_for_obclient.MEC_calc_time(D_MEC=task_MEC_all)
        return time_transmitting_calculating + time_MEC_calculating

    def simulation(self, R_client, v_x, v_y, x_client, y_client):
        """
        真实场景模拟
        :param R_client: 目标用户本地cpu计算速率
        :param v_x: 目标用户移动速度x分量
        :param v_y: 目标用户移动速度y分量
        :param x_client: 目标用户位置坐标x分量
        :param y_client: 目标用户位置坐标y分量
        :return: 总时延
        """
        #产生目标client和为其服务的MECserver
        global client_vector
        self._obclient_and_MECserver_for_obclient_producing(
            R_client=R_client, v_x=v_x, v_y=v_y, x_client=x_client, y_client=y_client)
        #判断目标client是否处于MECserver边缘
        obclient_pos_judge = self.__MECserver_for_obclient.client_pos_to_MECserver(self.__obclient)
        if obclient_pos_judge == 1: #目标client处于MECserver非边缘处
            #目标client获得MECserver服务范围内所有其它client的位置和速度信息
            client_vector = self.__MECserver_for_obclient.client_vector
            #生成计算任务#函数内部需要改
            self.__obclient.D_vector = client_vector

        elif obclient_pos_judge == 0: #目标client处于MECserver边缘处
            #所有MECserver将自己服务范围内的所有client位置和速度信息发送至Centerserver
            client_vector_to_Centerserver = []
            for mecserver in self.__MECserver_vector:
                # print(len(mecserver.client_vector))
                client_vector_to_Centerserver.extend(mecserver.client_vector)
            # print(client_vector_to_Centerserver[0])
            self.__CenterMECserver.client_vector = client_vector_to_Centerserver
            client_vector = self.__CenterMECserver.filter_client_vector(self.__obclient) #client_vector中有重复的需要略去

            self.__obclient.D_vector = client_vector
        return len(client_vector) if not obclient_pos_judge else 0 #打印中心center筛选出来的用户数

    def time_total_calculating(self, alphas):
        """
        计算总时延
        :param alphas: 子任务权值分配向量
        :return: 总时延
        """
        #本地计算时间
        time_local_calculating = self.__obclient.local_calc_time(alphas=alphas)
        #计算任务卸载时间和MECserver计算时间
        time_transmitting_and_MEC_calculating = self.time_transmitting_and_MEC_calculating(alphas=alphas)
        #总时延
        time_total = np.max(np.array([time_local_calculating, time_transmitting_and_MEC_calculating]))
        # print(type(time_total))
        return time_total

    def solve_problem(self, R_client, v_x, v_y, x_client, y_client, op_function='text'):
        """
        :param R_client: 目标用户本地cpu计算速率
        :param v_x: 目标用户移动速度x分量
        :param v_y: 目标用户移动速度y分量
        :param x_client: 目标用户位置坐标x分量
        :param y_client: 目标用户位置坐标y分量
        :param op_function: str, 优化方法名称，默认为'text'为输出0，'latency'为直接输出时延，'method'为根据method名进行优化
        :param T_TH: 对总时延的约束
        :return: 时延
        """
        #client_vector是由中心服务器筛选出来的用户数量
        client_vector_ = self.simulation(R_client=R_client, v_x=v_x, v_y=v_y, x_client=x_client, y_client=y_client)
        # if client_vector_:
        #     print('中心服务器筛选出来的用户数量为 %s' % client_vector_)
        # alphas = Map.param_tensor(param_range=(0, 1), param_size=[1, self.__client_num - 1])
        alphas = Map.param_tensor(param_range=(0, 1), param_size=[1, len(self.__obclient.D_vector)])
        def fun(alphas):
            """
            优化所需函数
            :param alphas: 目标client权值向量
            :return: 时延
            """
            time_all = self.time_total_calculating(alphas=alphas)
            return time_all

        #约束项函数
        # 约束条件 分为eq 和ineq
        # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
        cons = [{'type': 'ineq', 'fun':
            lambda alphas: self.__MECserver_for_obclient.Q_res() -  self.__obclient.task_distributing(alphas=alphas)},
                {'type': 'ineq', 'fun':
            lambda alphas: self.__obclient.Q_res() + self.__obclient.task_distributing(alphas=alphas) - np.sum(self.__obclient.D_vector)},
                {'type': 'ineq', 'fun':
            lambda alphas: self.__t_stay - self.time_transmitting_and_MEC_calculating(alphas=alphas)},
                # {'type': 'ineq', 'fun': lambda alphas: self.__T_epsilon - fun(alphas)},
                {'type': 'ineq', 'fun': lambda alphas: alphas.T},
                {'type': 'ineq', 'fun': lambda alphas: - alphas.T + 1}]

        # print(len(cons))
        if op_function == 'text':
            res = 0
        elif op_function == 'latency':
            res = fun(alphas=alphas)
        else:
            res = minimize(fun, alphas, method=op_function, constraints=cons, options={'maxiter':8}) #需要优化时打开

        return res



if __name__ == '__main__':
    pass
