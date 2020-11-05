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
    Qm_max = 1e3 * 500 #1000为用户数，实际可以考虑修改
    rng = np.random.RandomState(0)

    @staticmethod
    def next_state(loc, scale, limit_max:np.float):
        """

        :param loc:
        :param scale:
        :param limit_max:
        :return:
        """
        next_value = np.random.normal(loc=loc, scale=scale, size=1)
        if next_value[0] > loc:
            next_value = 2 * loc - next_value
        return min(next_value, limit_max)

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

    def _client_status_update2(self, clients_a:np.ndarray, clients_v:np.ndarray, clients_pos:np.ndarray):
        tau = 50 #单位为m,边界范围阈值，不同于服务器服务范围内边界，此处是另外一个定义
        theta_miu = 0.15
        miu = np.zeros(shape=(clients_v.shape[0], 1), dtype=np.float)
        sigma = 0.2
        x_l, x_r, y_l, y_r = 0, self._x_map, 0, self._y_map #道路网络边界
        acc_max = (20, 20) #反向加速度矢量
        #更新加速度
        #所有情况均需要的ou随机过程更新
        self._clients_a = self._clients_a + theta_miu * (miu - self._clients_a) + \
                          sigma * DynamicEnvironment.rng.normal(loc=0, scale=self._delta_t)
        #对x坐标处于边界范围的施加反向加速度
        need_to_add_a_left = clients_pos[0] > x_r - tau #处于右边界中
        need_to_add_a_right = clients_pos[0] < x_l + tau #处于左边界中
        if True in need_to_add_a_left:
            direct_none_guiyi_xl = clients_pos[need_to_add_a_left, 0] - x_l
            direct_guiyi_xl = direct_none_guiyi_xl / np.matmul(direct_none_guiyi_xl[np.newaxis, :], direct_none_guiyi_xl[:, np.newaxis])
            self._clients_a[need_to_add_a_left, 0] += acc_max[0] * direct_guiyi_xl
        if True in need_to_add_a_right:
            direct_none_guiyi_xr = clients_pos[need_to_add_a_right] - x_r
            direct_guiyi_xr = direct_none_guiyi_xr / np.matmul(direct_none_guiyi_xr[np.newaxis, :], direct_none_guiyi_xr[:, np.newaxis])
            self._clients_a[need_to_add_a_right, 0] += acc_max[0] * direct_guiyi_xr

        # 对x坐标处于边界范围的施加反向加速度
        need_to_add_a_down = clients_pos[-1] > y_r - tau  # 处于上边界中
        need_to_add_a_up = clients_pos[-1] < y_l + tau  # 处于下边界中
        if True in need_to_add_a_down:
            direct_none_guiyi_yl = clients_pos[need_to_add_a_down, -1] - y_l
            direct_guiyi_yl = direct_none_guiyi_yl / np.matmul(direct_none_guiyi_yl[np.newaxis, :],
                                                               direct_none_guiyi_yl[:, np.newaxis])
            self._clients_a[need_to_add_a_down, 0] += acc_max[-1] * direct_guiyi_yl
        if True in need_to_add_a_up:
            direct_none_guiyi_yr = clients_pos[need_to_add_a_up] - y_r
            direct_guiyi_yr = direct_none_guiyi_yr / np.matmul(direct_none_guiyi_yr[np.newaxis, :],
                                                               direct_none_guiyi_yr[:, np.newaxis])
            self._clients_a[need_to_add_a_up, -1] += acc_max[-1] * direct_guiyi_yr

        #更新速度
        clients_v = clients_v + self._clients_a * self._delta_t
        #不得超过限速阈值
        clients_v = np.where(clients_v>DynamicEnvironment.vxy_max, DynamicEnvironment.vxy_max, clients_v)
        #更新位置
        clients_pos = clients_v + clients_v * self._delta_t
        return clients_v, clients_pos

    def __init__(self):
        self._x_map = 1e5
        self._y_map = 1e5
        self._client_num = 200
        self._MECserver_num = 4
        self._R_client_mean = 1e3 #HZ
        self._R_MEC_mean = 1e5 #Hz  #单个计算任务量均值在1000bit
        self._vxy_client_range = (-60, 60)
        self._T_epsilon = 1 #s
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
        # self._iter_time = 0 #迭代计数
        # self._iteration_time = iteration_time #迭代总次数

    def change_environment(self):
        """
        动态改变环境
        :return:
        v0(0): obclient移动速度矢量
        pos0(1): obclient位置坐标
        Rc(2): obclient本地计算速率
        Rm(3): MEC端任务计算速率
        """
        self._delta_t = 1 #动态环境更新时间
        next_state_func = np.frompyfunc(DynamicEnvironment.next_state, 3, 1)
        #历史速度
        clients_v = self.map.clients_v
        #################直接对v从正态分布中采样更新################
        # # 将移动速度在当前值基础上增减, Dx=21
        # clients_v_new = np.asarray(next_state_func(clients_v, 21), 'float32')
        #
        # #除目标client之外其它client的位置更新
        # clients_v_new, clients_pos_new = self._client_status_update(clients_v=clients_v_new, clients_pos=self.map.clients_pos)

        #################用ou随机过程更新加速度，进而更新v###################
        a_init = 2 #m/s^2 加速度初始值
        d_a_init = 0.5 #加速度初始值方差
        if not hasattr(self, '_clients_a'):
            self._clients_a = DynamicEnvironment.rng.normal(loc=a_init, scale=d_a_init)
        clients_v_new, clients_pos_new = self._client_status_update2(clients_a=self._clients_a,
                                                                     clients_v=clients_v,
                                                                     clients_pos=self.map.clients_pos)
        #将本地计算速率在当前值基础上增减, Dx=100
        #R_t*delta_t-yita表示在t-1时刻R_t值的基础上减一个小量确保不越界
        R_client = self.map.clients_R_client
        yita = 0.01
        #方差
        rou_R_client = 100
        R_client_new = np.asarray(next_state_func(R_client*(1-yita), rou_R_client, DynamicEnvironment.Rc_max), 'float32')
        #将云端计算速率在当前值基础上增减, Dx=6000
        #方差
        rou_R_mec = 6000
        mecservers_R_MEC = self.map.MECservers_R_MEC
        mecservers_R_MEC_new = np.asarray(next_state_func(mecservers_R_MEC*(1-yita), rou_R_mec, DynamicEnvironment.Rc_max), 'float64')
        #将云端任务量存储阈值在当前那值基础上增减, Dx=10000
        #方差
        rou_q_mec = 10000
        q_MEC = self.map.Q_MEC
        D_pre_mec = np.sum(self.map.ob_client.D_vector * self.get_alpha())
        q_MEC_new = np.asarray(next_state_func(q_MEC-yita*D_pre_mec, rou_q_mec, DynamicEnvironment.Qm_max), 'float32')
        #除目标用户外的其它用户属性更新
        self.map.client_vector = (R_client_new, clients_v_new, clients_pos_new[:, 0], clients_pos_new[:, -1])
        # print('client_vector is finished')
        #得到上一时刻的计算任务量D，在本次计算任务中减去确保，假定上次任务量在下一秒更新时有残留
        D_pre_c = np.sum(self.map.ob_client.D_vector * (1 - self.get_alpha()))
        #更新MEC服务器
        self.map.mecserver_vector = (mecservers_R_MEC_new, q_MEC_new)
        # print('mecserver_vector is finished')
        #将本地任务量存储阈值在当前值基础上增减, DX=15
        #方差
        rou_q_c = 15
        obclient_Q_client = self.map.ob_client.Q_client
        obclient_Q_client_new = np.asarray(next_state_func(obclient_Q_client-yita*D_pre_c, rou_q_c, DynamicEnvironment.Qc_max), 'float64')[0]
        #获取目标client的属性，计算其更新属性
        obclient = self.map.ob_client
        r_client = obclient.R_client
        #方差
        rou_r_c = 100
        r_client_new = np.asarray(next_state_func(r_client, rou_r_c, DynamicEnvironment.Rc_max), 'float32')[0]
        # print('r', r_client_new, r_client_new[0])
        v = obclient.v
        axis = obclient.axis
        #################直接对v从正态分布中采样更新################
        # v_new = np.asarray(next_state_func(v, 21), 'float32')
        # obclient_v_new, obclient_pos_new = self._client_status_update(clients_v=np.array(v_new)[np.newaxis, :],
        #                                                               clients_pos=np.array(axis)[np.newaxis, :])
        #################用ou随机过程更新加速度，进而更新v###################
        if not hasattr(self, '_obclient_a'):
            self._obclient_a = DynamicEnvironment.rng.normal(loc=a_init, scale=d_a_init)
        obclient_v_new, obclient_pos_new = self._client_status_update2(clients_a=self._obclient_a,
                                                                       clients_v=np.array(v)[np.newaxis, :],
                                                                       clients_pos=np.array(axis)[np.newaxis, :])
        #latency
        self.map.ob_client.Q_client = obclient_Q_client_new
        return r_client_new, obclient_v_new, obclient_pos_new, mecservers_R_MEC_new.mean(), \
               obclient.Q_res(), self.map.mecserver_for_obclient.Q_res(), obclient.D_vector

    def get_alpha(self):
        """

        :return:
        """
        return self.map.ob_client.alpha_vector

    def get_latency(self, r_client_new, obclient_v_new, obclient_pos_new, alphas:np.ndarray=None, op_function='latency'):
        """

        :return:
        """
        client_Q_constraint, mec_Q_constraint, t_constraint, latency = self.map.solve_problem(R_client=r_client_new,
                                                                            v_x=obclient_v_new[0],
                                                                            v_y=obclient_v_new[-1],
                                                                            x_client=obclient_pos_new[0],
                                                                            y_client=obclient_pos_new[-1],
                                                                            op_function=op_function,
                                                                            alphas=alphas)
        return client_Q_constraint, mec_Q_constraint, t_constraint, latency

    def __enter__(self):
        rng = np.random.RandomState(0)
        r_client_new = rng.normal(loc=1e3, scale=1, size=1)[0] #目标用户初始计算速度均值
        obclient_v_new = (10, 10)
        # obclient_pos_new = (97779.54569559613, 88473.93449231013)
        obclient_pos_new = (25000., 25000.)
        self.get_latency(r_client_new=r_client_new, obclient_v_new=obclient_v_new, obclient_pos_new=obclient_pos_new)
        return self

    def s_calc(self):
        """
        计算状态向量
        :return:
        """
        r_obclient, v_obclient, pos_obclient, r_mec, Q_c, Q_m, D_vector = self.change_environment()
        # 限制alphas的取值范围在0~1之间的函数
        f = np.frompyfunc(lambda x: min(1, max(0, x)), 1, 1)
        s_pre = np.array(
            [v_obclient.ravel()[0], v_obclient.ravel()[-1], pos_obclient.ravel()[0], pos_obclient.ravel()[-1],
             r_obclient, Q_c/100, r_mec, Q_m/1000])
        alphas_prune = np.asarray(f(self.get_alpha()), 'float64')[:, np.newaxis]
        s_suf = np.hstack((D_vector * alphas_prune, D_vector * (1 - alphas_prune))).ravel()
        # 合并总状态向量
        s = np.hstack((s_pre, s_suf))[np.newaxis, :]
        # s[0, 4] = s[0, 4][0]
        # s[0, 5] = s[0, 5][0]
        # print(s.__class__)
        return s

    def reset(self):
        """
        状态初始化
        :return:
        """
        return self.s_calc()

    def step(self, alphas):
        """
        根据动作更新状态，并输出状态和对应奖励
        :param alphas:
        :return: s(shape=(1, 136)), r
        """
        #更新状态向量
        s = self.s_calc()
        # print(s)
        r_client_new, obclient_v_new, obclient_pos_new = s[:, 4], s[:, 0:2], s[:, 2:4]
        obclient_v_new = obclient_v_new.ravel().tolist()
        obclient_pos_new = obclient_pos_new.ravel().tolist()
        self.map.alphas = alphas
        Q_c_local, Q_m_mec, t_constraint, latency = self.get_latency(r_client_new=r_client_new, obclient_v_new=obclient_v_new,
                                                       obclient_pos_new=obclient_pos_new)
        # print('latency %s' %latency)
        #对应新奖励
        belta1 = 0.99
        belta2 = 0.99
        belta3 = 0.99
        limit_fun = lambda x: -2. if x < 0. else 0
        # r = -float(latency)
        #最值限制
        r = -float(latency) + belta1 * limit_fun(min(0., float(Q_c_local))) + belta2 * \
            limit_fun(min(0., float(Q_m_mec))) + belta3 * limit_fun(min(0., float(t_constraint)))
        #sigmoid
        sigmoid = lambda x: 1 / (1 + np.exp(-x)) - 1
        # r = -float(latency) + belta1 * sigmoid(min(0., float(Q_c_local))) + belta2 * \
        #     sigmoid(min(0., float(Q_m_mec))) + belta3 * sigmoid(min(0., float(t_constraint)))
        #tanh
        tanh = lambda x: (np.exp(float(x)) - np.exp(-float(x))) / (np.exp(float(x)) + np.exp(-float(x))+3)
        # r = -float(latency) + belta1 * tanh(min(0., float(Q_c_local))) + belta2 * \
        #     tanh(min(0., float(Q_m_mec))) + belta3 * tanh(min(0., float(t_constraint)))

        return s, float(r), float(latency)

    def step_gibbs_sampling(self, alphas):
        """
        适用于吉布斯采样的状态更新
        :param alphas: 状态向量
        :return:
        """
        # 更新状态向量
        # self.change_environment()
        self.map.alphas = alphas
        _, _, _, latency = self.get_latency(r_client_new=self.map.ob_client.R_client,
                                            obclient_v_new=self.map.ob_client.v,
                                            obclient_pos_new=self.map.ob_client.axis)
        return latency

    def get_some_param(self):
        """
        返回当前时刻目标用户的Qc, MEC的Qm, 传输速率Ru, MEC的Rm, ts, D
        :return: Qc, Qm, Ru, Rm, ts, D
        """
        return self.map.ob_client.Q_client, self.map.mecserver_for_obclient, self.map.R_transmit, \
               self.map.MECservers_R_MEC, self.map.t_stay, self.map.ob_client.D_vector

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == '__main__':
    with DynamicEnvironment() as d:
        for episode in range(100):
            s0 = d.reset()
            # print('s0', type(s0)==np.object)
            for step in range(4):
                a0 = np.random.normal(size=(1, 64))
                s1, r1, latency = d.step(alphas=a0)
                print(s0.shape, a0.shape, r1, s1.shape)
        # s0 = d.reset()
        # s1, r1 = d.step(alphas=np.random.normal(size=(1, 64)))
        # print(s0)
        # print('\n')
        # print(s0[:, :8], '\n\n', s0[:, 8:])
        # print(r1, type(r1))

