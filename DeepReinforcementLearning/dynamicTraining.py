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
            # print(type(s0[0][0]), s0.shape)
            episode_reward = 0 #总奖励
            for step in range(4):
                a0 = agent.act(s=s0)
                s1, r1 = d.step(alphas=a0)
                # print(s0.shape, a0.shape)
                agent.store_transition(s=s0, a=a0, r=r1, s_=s1)
                episode_reward += r1
                s0 = s1
                agent.learn()
            print('episode {0}: {1:.4}'.format(episode, episode_reward))

if __name__ == '__main__':
    trainingAndOptimization()