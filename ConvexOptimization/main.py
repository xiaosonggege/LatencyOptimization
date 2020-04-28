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
from RoadNetwork.Environment import Map
from ConvexOptimization.multiprocessingfunc import datagenerator
import multiprocessing
import psutil
from matplotlib import pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.ticker import FuncFormatter

def main_function(vxy_client_range=(-60, 60), T_epsilon=8*60, client_num=1000, B=6.3e+6, plotfun=None):
    """
    测试函数
    :param vxy_client_range: 移动用户速度各分量范围值
    :param T_epsilon: 时间阈值
    :param client_num: 移动用户数量
    :param B: 无线信道带宽
    :param plotfun: 绘图函数
    :return: 最优延时
    """
    x_map = 1e5
    y_map = 1e5
    client_num = client_num
    MECserver_num = 4
    R_client_mean = 1e3 #HZ
    R_MEC_mean = 1e5 #Hz  #单个计算任务量均值在1000bit
    vxy_client_range = vxy_client_range #(-15, 15)
    T_epsilon = T_epsilon #5 * 60
    Q_client = 1e2
    Q_MEC = 1e3 * client_num  # 够承载10000用户所有计算任务的
    MECserver_num_sqrt = np.sqrt(MECserver_num)
    server_r = 1 / np.sqrt(2*MECserver_num) * x_map #(x_map / (MECserver_num_sqrt +1)) * np.sqrt(2) #边缘服务器间等距且与边界等距
    r_edge_th = server_r * (2 - np.sqrt(2)) #x_map / (MECserver_num_sqrt +1)
    B = B #6.3e+6
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
        x_client=25000., #97779.54569559613, #Map.rng.uniform(low=0, high=x_map),
        y_client=25000., #88473.93449231013, #Map.rng.uniform(low=0, high=y_map),
        op_function='slsqp'
    )

    print(res)
    print('最优时延结果为: %s' % res.fun)
    print('取得最优时延时优化参数向量为:\n', res.x)
    print('client={0}时迭代次数为: {1}'.format(client_num, res.nit))
    print('迭代成功？ %s' % res.success)

    #画图专区
    # print(map.clients_pos)
    # print(type(map.clients_pos), map.clients_pos.shape)
    # print(map.MECserver_for_obclient)
    # print(map.Obclient)
    # print(map.MECserver_vector)
    # plotfun(map.clients_pos, map.Obclient, *map.MECserver_for_obclient)
    # plotfun(map.clients_pos, map.Obclient, *map.MECserver_vector)
    # return 0
    return res.fun


if __name__ == '__main__':
    vxy_client_range = None
    T_epsilon = None
    client_num = 200
    B = None
    print('正在执行优化')
    main_function()
    print('执行完成')

#########################绘图####################################
    # for T_epsilon in [e * 60 for e in range(1, 11)]:
    #     main_function(T_epsilon=T_epsilon, vxy_client_range=(-30, 30))
    # for client_num in [e for e in range(1000, 7700, 700)]:
    #     main_function(client_num=client_num)
    def plotfun0(clients_pos:list, obclient_pos:tuple, *MECSever:tuple)->None:
        """
        绘制边缘服务器与目标client示意图
        :param clients_pos: 所有用户坐标
        :param obclient_pos: 目标用户坐标
        :param MECSever: 与目标用户直接进行通信的边缘服务器信息
        :return: None
        """
        clxs, clys = np.split(ary=np.array(clients_pos), indices_or_sections=2, axis=1)
        obclx, obcly = np.split(ary=np.array(obclient_pos), indices_or_sections=2)
        MECServer_axis, r, r_TH = MECSever
        # print(r, r_TH)
        fig, ax = plt.subplots()
        ax.scatter(x=clxs, y=clys, s=6, label='mobile users')
        ax.scatter(x=obclx, y=obcly, s=19, c='r', label='target user')
        ax.scatter(x=MECServer_axis[0], y=MECServer_axis[-1], c='m', label='MECServer')
        #画圆区域
        x = np.arange(MECServer_axis[0]-r, 1e5, 0.1)
        print(x[-1])
        y_up = MECServer_axis[-1] + np.sqrt(r**2-(x-MECServer_axis[0])**2)
        y_down = MECServer_axis[-1] - np.sqrt(r**2-(x-MECServer_axis[0])**2)
        ax.plot(x, y_up, c='y', label='UpperGround')
        ax.plot(x, y_down, c='y')
        x_TH = np.arange(MECServer_axis[0]-r_TH, MECServer_axis[0]+r_TH, 0.1)
        yTH_up = MECServer_axis[-1] + np.sqrt(r_TH**2-(x_TH-MECServer_axis[0])**2)
        yTH_down = MECServer_axis[-1] - np.sqrt(r_TH ** 2 - (x_TH - MECServer_axis[0]) ** 2)
        ax.plot(x_TH, yTH_up, c='g', label='LowerRange')
        ax.plot(x_TH, yTH_down, c='g')
        plt.legend()

        def formatnum(x, pos):
            return '$%.1f$x$10^{5}$' % (x / 100000)

        formatter = FuncFormatter(formatnum)
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.show()
    # main_function(plotfun=plotfun0)

    def plotfun1(clients_pos:list, obclient_pos:tuple, *MECSevers:tuple)->None:
        """"""
        print('进来了')
        clxs, clys = np.split(ary=np.array(clients_pos), indices_or_sections=2, axis=1)
        obclx, obcly = np.split(ary=np.array(obclient_pos), indices_or_sections=2)
        MECServer_posx, MECServer_posy, r, r_TH = MECSevers
        # print(MECServer_posx, '\n', MECServer_posy)

        circle_up = lambda x, y, r_TH, x_TH: y + np.sqrt(r_TH**2-(x_TH-x)**2)
        circle_down = lambda x, y, r_TH, x_TH: y - np.sqrt(r_TH ** 2 - (x_TH - x) ** 2)
        fig = plt.figure('完整的道路网络图')
        ax = fig.add_subplot(1, 1, 1)
        flag = 1
        for pos in zip(MECServer_posx, MECServer_posy):
            x_ = np.arange(pos[0] - r, pos[0] + r, 0.1)
            x_TH = np.arange(pos[0] - r_TH, pos[0] + r_TH, 0.1)
            y_up = circle_up(x=pos[0], y=pos[-1], r_TH=r, x_TH=x_)
            y_down = circle_down(x=pos[0], y=pos[-1], r_TH=r, x_TH=x_)
            y_THup = circle_up(x=pos[0], y=pos[-1], r_TH=r_TH, x_TH=x_TH)
            y_THdown = circle_down(x=pos[0], y=pos[-1], r_TH=r_TH, x_TH=x_TH)
            if flag:
                ax.plot(x_, y_up, c='y', label='The boundary of the serving area')
            else:
                ax.plot(x_, y_up, c='y')
            ax.plot(x_, y_down, c='y')
            if flag:
                ax.plot(x_TH, y_THup, c='g', label='Inner boundary of the border region')
            else:
                ax.plot(x_TH, y_THup, c='g')
            ax.plot(x_TH, y_THdown, c='g')
            if flag:
                ax.scatter(x=pos[0], y=pos[-1], c='m', label='Edge servers')
                flag = 0
            ax.scatter(x=pos[0], y=pos[-1], c='m')
        ax.scatter(x=clxs, y=clys, s=6, label='Mobile users')
        # ax.scatter(x=obclx, y=obcly, c='r', label='target user')

        def formatnum(x, pos):
            return '$%.1f$x$10^{5}$' % (x / 100000)

        formatter = FuncFormatter(formatnum)
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.legend(loc='upper left')
        fig.show()
    # main_function(plotfun=plotfun1)

    def plotfun2():
        T_e = [e * 60 for e in range(1, 11)]
        #Vmax=120
        clients_choice_num1 = [4, 27, 46, 102, 106, 144, 188, 258, 278, 336]
        #Vmax=90
        clients_choice_num2 = [2, 16, 30, 64, 73, 89, 125, 183, 190, 218]
        #Vmax=60
        clients_choice_num3 = [2, 11, 20, 43, 45, 53, 78, 105, 117, 127]
        fig, ax = plt.subplots()
        ax.plot(T_e, clients_choice_num1, c='r', marker='<', label=r'$V_\max=120km/h$')
        ax.plot(T_e, clients_choice_num2, c='g', marker='*', label=r'$V_\max=90km/h$')
        ax.plot(T_e, clients_choice_num3, c='b', marker='>', label=r'$V_\max=60km/h$')
        plt.legend()
        plt.xlabel(r'$T_\epsilon/s$')
        plt.ylabel('Number of mobile users filtered by the central server')
        plt.grid(axis='x', linestyle='-.')
        plt.grid(axis='y', linestyle='-.')
        plt.show()
    # plotfun2()
    def plotfun3():
        client_nums = [e for e in range(1000, 7301, 700)]
        Latencys = [2.206, 3.771, 4.634, 7.001, 8.957, 10.532, 12.137, 14.384, 15.779, 28.947]
        client_nums_choice = [220, 391, 583, 758, 913, 1069, 1295, 1436, 1627, 1779]
        fig, ax = plt.subplots()
        ax.plot(np.array(client_nums), Latencys, c='r', marker='.', label='Optimization success')
        ax.scatter(x=np.array([client_nums[-1]]), y=Latencys[-1], c='g', s=100, label='Optimization failed')
        plt.xlabel('Client Number')
        plt.ylabel('Latency-time/s')
        # plt.title('Line chart of total latency over number of mobile users')
        plt.grid(axis='x', linestyle='-.')
        plt.grid(axis='y', linestyle='-.')
        plt.legend()
        plt.show()
    # plotfun3()
#########################带宽与时延关系图##########################
    # import re
    # from matplotlib import ticker as mtick
    # data = []
    # data_all = []
    # regex = re.compile(pattern='\d+.\d+')
    # for i in [700, 2000, 7000]:
    #     with open(file=r'/Users/songyunlong/Desktop/实验室/时延模型/file_%s.txt' % str(i), mode='r') as f:
    #         while True:
    #             line = f.readline()
    #             if line:
    #                 tuplist = regex.findall(line)
    #                 if len(tuplist):
    #                     data.append([float(i) for i in tuplist])
    #             else:
    #                 break
    #     data = sorted(data, key=lambda x: x[0])
    #     import copy
    #     data_all.append([tup[-1] for tup in data])
    #     data.clear()
    # # for data in data_all:
    # #     print(data)
    # data_all = np.array(data_all)
    # print(data_all[0, 0])
    # fig, ax = plt.subplots()
    # x = np.linspace(5.2, 7, 10)*1e6
    # ax.plot(x, data_all[0], c='r', marker='.', label='client number=700')
    # ax.plot(x, data_all[1], c='g', marker='>', label='client number=2000')
    # ax.plot(x, data_all[2], c='b', marker='^', label='client number=7000')
    # ax.set_xlabel('Bandwidth/MHz')
    # ax.set_ylabel('Latency-time/s')
    # ax.legend(loc='upper left')
    #
    # formatnumx = lambda x, pos: '$%.1f$x$10^{6}$' % (x / 1000000)
    # formatterx = FuncFormatter(formatnumx)
    # ax.xaxis.set_major_formatter(formatterx)
    # fmt = lambda x, pos:'$%.2f$' % (x)
    # yticks = FuncFormatter(fmt)
    # ax.yaxis.set_major_formatter(yticks)
    # ax.grid(axis='x', linestyle='-.')
    # ax.grid(axis='y', linestyle='-.')
    # ax.set_ylim((0, 25))
    # fig.show()
#########################时延下降模型##############################
    import pandas as pd
    def ploting():
        fig, ax = plt.subplots()
        x = np.linspace(1, 8, 8)
        y_matrix = pd.DataFrame(
            data=np.array([
                [9.37, 0.1971, 0.1896, 0.1801, 0.1801, 0, 0, 0],
                [19.12, 0.4164, 0.4103, 0.4021, 0.4021, 0, 0, 0],
                [80.18, 1.6337, 1.6316, 1.613, 1.61, 0, 0, 0],
                [113.03, 2.2190, 2.2179, 2.2084, 2.2033, 2.201, 2.20, 2.20],
                [134.56, 2.8222, 2.8101, 2.8101, 0, 0, 0, 0]
            ]).T,
            columns=[100, 200, 600, 1000, 1200]
        )
        # print(y_matrix)
        ax.plot(x[:-3], y_matrix[100][:-3], c='r', marker='.', label='client number=100')
        ax.plot(x[:-3], y_matrix[200][:-3], c='g', marker='>', label='client number=200')
        ax.plot(x[:-3], y_matrix[600][:-3], c='b', marker='<', label='client number=600')
        ax.plot(x, y_matrix[1000], c='y', marker='^', label='client number=1000')
        ax.plot(x[:-4], y_matrix[1200][:-4], c='m', marker='o', label='client number=1200')
        ax.set_xlabel('Iterations/times')
        ax.set_ylabel('Latency-time/s')
        ax.legend(loc='upper right')
        ax.grid(axis='x', linestyle='-.')
        ax.grid(axis='y', linestyle='-.')
        # ax.set_ylim(0, 25)
        fig.show()
    # ploting()
#########################多进程生成数据############################
    # import sys
    # print('开始执行多进程')
    # dg = datagenerator(func=main_function, client_num=int(sys.argv[1]))
    # # # dg.name('vxy_client_range', [(-e, e) for e in range(15, 65, 5)])
    # dg.name('client_num', [1000])
    # # dg.name('B', np.linspace(5.2, 7, 10)*1e6)
    # # # dg.name('T_epsilon', [e * 5 * 60 for e in range(1, 11)])
    # dg.multiprocess()
    # print('多进程结束')