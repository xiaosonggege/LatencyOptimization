#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: dynamicTraining
@time: 2020/4/22 7:30 下午
'''
from DeepReinforcementLearning.DynamicRoadNetwork import DynamicEnvironment
from DeepReinforcementLearning.DeepNetwork import Agent
import numpy as np
import torch
import time

def trainingAndOptimization():
    agent = Agent()
    with DynamicEnvironment() as d:
        for episode in range(100):
            s0 = d.reset()
            s0 = np.asarray(s0, 'float32')
            #添加slsqp对比实验位置
            # latency_slsqp = 0
            # print(type(s0[0][0]), s0.shape)
            episode_reward = 0 #总奖励
            latency_total = 0 #时延输出
            #测试时间
            start = time.time()
            for step in range(100):
                a0 = agent.act(s=s0)
                s1, r1, latency = d.step(alphas=a0)
                # print('info of reward', r1, type(r1), type(s1), type(s1[0]))
                # print(s0.shape, a0.shape)
                agent.store_transition(s=s0, a=a0, r=r1, s_=s1)
                episode_reward += r1
                latency_total += latency
                s0 = s1
                # 添加slsqp对比实验位置
                # r_client_new, obclient_v_new, obclient_pos_new = s0[:, 4], s0[:, 0:2], s0[:, 2:4]
                # latency_slsqp += d.get_latency(r_client_new=r_client_new, obclient_v_new=obclient_v_new.ravel().tolist(),
                #                                obclient_pos_new=obclient_pos_new.ravel().tolist(),
                #                                op_function='slsqp', alphas=a0)
                agent.learn()
            end = time.time()
            print('100轮时间为%s' % (end-start))
            data_str = 'episode {0} reward: {1:.4}, latency: {2:.4}'.format(episode, episode_reward/100, latency_total/100)
            # with open(file='/Users/songyunlong/Desktop/ddpg200-tanh.txt', mode='a') as f:
            #     f.write(data_str+'\n')
            print(data_str)
            # print('episode {0} slsqp latency: {1:.4}'.format(episode, latency_slsqp/4))

def gibbssampling():
    """
    用Gibbs抽样对得到各时刻的动作向量
    """
    rng = np.random.RandomState(0)
    def gibbs(alphas:np.ndarray, Qc:np.float, Qm:np.float, Ru:np.float, Rm:np.float, ts:np.float, D:np.ndarray):
        """
        Gibbs抽样
        :return:
        """
        for i in range(alphas.shape[0]):
            alpha_lower = 1 - (Qc - np.matmul(alphas[np.newaxis, :],D[:, np.newaxis]) + alphas[i]*D[i]) / D[i]
            lower = max(0, alpha_lower.ravel()[0])
            alpha_upper1 = (ts / (1 / Ru + 1 / Rm) - np.matmul(alphas[np.newaxis, :],D[:, np.newaxis]) +
                            alphas[i]*D[i]) / D[i]
            alpha_upper2 = (Qc - np.matmul(alphas[np.newaxis, :],D[:, np.newaxis]) + alphas[i]*D[i]) / D[i]
            upper = min(1, alpha_upper1.ravel()[0], alpha_upper2.ravel()[0])
            alphas[i] = rng.uniform(low=lower, high=upper)
        return alphas

    with DynamicEnvironment() as d:
        average_latencys = None #记录每个episode的整条采样链的时延均值
        for episode in range(100):
            latency_total = 0 #时延输出
            #测试时间
            # start = time.time()
            for step in range(4):
                Qc, Qm, Ru, Rm, ts, D = d.get_some_param()
                alphas = np.zeros(shape=D.shape)
                alphas = gibbs(alphas=alphas, Qc=Qc, Qm=Qm, Ru=Ru, Rm=Rm, ts=ts, D=D)
                latency = d.step_gibbs_sampling(alphas=alphas)
                latency_total += latency
            # end = time.time()
            # print('100轮时间为%s' % (end-start))
            average_latencys = np.array([[latency_total/4]]) if average_latencys is None else\
                np.hstack((average_latencys, np.array([[latency_total/4]])))
            data_str = 'episode {0}, latency: {1:.4}'.format(episode, latency_total/4)
            data_str_2 = ' '.join([str(e) for e in average_latencys.ravel().tolist()])
            # with open(file='/Users/songyunlong/Desktop/gibbs.txt', mode='w') as f:
            #     f.write(data_str_2)
            print(data_str)

if __name__ == '__main__':
    # trainingAndOptimization()
    gibbssampling()