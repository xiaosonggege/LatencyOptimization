#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: main
@time: 2020/1/8 4:00 下午
'''
import numpy as np
from Environment import Map

def main():
    """
    测试函数
    :return: None
    """
    x_map = 1e5
    y_map = 1e5
    client_num = 1000
    MECserver_num = 4
    R_client_mean = 1e3
    R_MEC_mean = 1e5
    vxy_client_range = (-15, 15)
    T_epsilon = 5 * 60
    Q_client = 1e2
    Q_MEC = 1e3 * client_num  # 够承载10000用户所有计算任务的
    MECserver_num_sqrt = np.sqrt(MECserver_num)
    server_r = (x_map / (MECserver_num_sqrt +1)) * np.sqrt(2)
    r_edge_th = x_map / (MECserver_num_sqrt +1)
    B = 6.3e+6
    N0 = 1e-10
    P = 1e-6
    h = 0.95
    delta = -0.9

    map = Map(
        x_map=x_map,
        y_map=y_map,
        client_num=client_num,
        MECserver_num=MECserver_num,
        R_client_mean=R_client_mean,
        R_MEC_mean=R_MEC_mean,
        vxy_client_range=vxy_client_range,
        T_epsilon=T_epsilon,
        Q_client=Q_client,
        Q_MEC=Q_MEC, #够承载10000用户所有计算任务的
        server_r=server_r,
        r_edge_th=r_edge_th,
        B=B,
        N0=N0,
        P=P,
        h=h,
        delta=delta
    )

    res = map.solve_problem(
        R_client=Map.param_tensor_gaussian(mean=R_client_mean, var=1, param_size=1),
        v_x=10,
        v_y=10,
        x_client=Map.rng.uniform(low=0, high=x_map),
        y_client=Map.rng.uniform(low=0, high=y_map),
        op_function='SLSQP'
    )
    # print('最优时延结果为: %s' % res.fun)
    # print('取得最优时延时优化参数向量为:\n', res.x)
    # print('迭代次数为: %s' % res.nit)
    # print('迭代成功？ %s' % res.success)


if __name__ == '__main__':
    main()
    # a = 1000
    # print(1e3 * a)