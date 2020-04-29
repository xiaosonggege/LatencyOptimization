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
            for step in range(4):
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
            data_str = 'episode {0} reward: {1:.4}, latency: {2:.4}'.format(episode, episode_reward/4, latency_total/4)
            with open(file='/Users/songyunlong/Desktop/ddpg200-tanh.txt', mode='a') as f:
                f.write(data_str+'\n')
            print(data_str)
            # print('episode {0} slsqp latency: {1:.4}'.format(episode, latency_slsqp/4))
if __name__ == '__main__':
    trainingAndOptimization()