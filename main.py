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
    map = Map(
        x_map=0,
        y_map=0,
        client_num=0,
        MECserver_num=0,
        R_client_mean=0,
        R_MEC_mean=0,
        vxy_client_range=(0, 1),
        T_epsilon=0,
        Q_client=0,
        Q_MEC=0,
        server_r=0,
        r_edge_th=0,
        B=0,
        N0=0,
        P=0,
        h=0,
        delta=0
    )
    res = map.solve_problem(
        R_client=0,
        v_x=0,
        v_y=0,
        x_client=0,
        y_client=0,
        op_function=0
    )
    print('最优时延结果为: %s' % res.fun)
    print('取得最优时延时优化参数向量为:\n', res.x)
    print('迭代次数为: %s' % res.nit)
    print('迭代成功？ %s' % res.success)


if __name__ == '__main__':
    main()