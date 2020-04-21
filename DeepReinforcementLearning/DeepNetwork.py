#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: DeepNetwork
@time: 2020/4/21 3:46 下午
'''
import torch
from torch.nn import functional as F
import numpy as np
from functools import reduce

class Actor(torch.nn.Module):
    def __init__(self, s1_dim, s2_dim, a_dim):
        """

        :param s1_dim: 6
        :param s2_dim: 128
        :param a_dim: 64
        """
        super().__init__()
        self._s2linear1 = torch.nn.Linear(in_features=s2_dim, out_features=100)
        self._s2net_sequential1 = torch.nn.Sequential()
        self._s2net_sequential1.add_module(name='conv1',
                                         module=torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3))
        self._s2net_sequential1.add_module(name='relu1', module=torch.nn.ReLU())
        self._s2net_sequential1.add_module(name='conv2',
                                         module=torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2))
        self._s2net_sequential1.add_module(name='relu2', module=torch.nn.ReLU())

        self._s1linear1 = torch.nn.Sequential(torch.nn.Linear(in_features=s1_dim, out_features=28),
                                              torch.nn.Linear(in_features=28, out_features=28))
        self._combinelinear = torch.nn.Linear(in_features=128, out_features=a_dim)

    def forward(self, *input):
        """

        :param input:
        :return:
        """
        s1, s2 = input #输入维度为(batch_size, 6),(batch_size, 1, 64, 2)
        s2 = s2.reshape(shape=(-1, 1, reduce(lambda x, y: x*y, s2.size[1:])))
        s1net = self._s1linear1(s1)
        s1net = F.relu(input=s1net)
        s2net = self._s2linear1(s2)
        s2net = F.relu(input=s2net)
        s2net = self._s2net_sequential1(s2net)
        s2net = s2net.reshape(shape=[-1, reduce(lambda x, y:x*y, s2net.size()[1:])])
        a = self._combinelinear(torch.cat(tensors=[s1net, s2net], dim=1))
        a = F.tanh(input=a)
        return a

class Critic(torch.nn.Module):
    def __init__(self, s1_dim, s2_dim, a_dim):
        super().__init__()
        self._s2linear1 = torch.nn.Linear(in_features=s2_dim, out_features=100)
        self._s2net_sequential1 = torch.nn.Sequential()
        self._s2net_sequential1.add_module(name='conv1',
                                           module=torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7,
                                                                  padding=3))
        self._s2net_sequential1.add_module(name='relu1', module=torch.nn.ReLU())
        self._s2net_sequential1.add_module(name='conv2',
                                           module=torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5,
                                                                  padding=2))
        self._s2net_sequential1.add_module(name='relu2', module=torch.nn.ReLU())

        self._s1linear1 = torch.nn.Sequential(torch.nn.Linear(in_features=s1_dim, out_features=28),
                                              torch.nn.Linear(in_features=28, out_features=28))
        self._anet = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128))
        self._combinelinear = torch.nn.Linear(in_features=128+128, out_features=1)

    def forward(self, *input):
        """

        :param input: s1, s2, a
        :return:
        """
        s1, s2, a = input  # 输入维度为(batch_size, 6),(batch_size, 1, 64, 2),(batch_size, 1, 64)
        s2 = s2.reshape(shape=(-1, 1, reduce(lambda x, y: x * y, s2.size[1:])))
        s1net = self._s1linear1(s1)
        s1net = F.relu(input=s1net)
        s2net = self._s2linear1(s2)
        s2net = F.relu(input=s2net)
        s2net = self._s2net_sequential1(s2net)
        s2net = s2net.reshape(shape=[-1, reduce(lambda x, y: x * y, s2net.size()[1:])])
        anet = self._anet(a)
        q = self._combinelinear(torch.cat(tensors=[s1net, s2net, anet], dim=1))
        q = F.tanh(input=q)
        return a

class Agent:
    def __init__(self):
        self._s1_dim = 6
        self._s2_dim = (64, 2)
        self._a_dim = 64
        self.actor_eval = Actor(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.actor_target = Actor(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.critic_eval = Critic(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.critic_target = Critic(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.actor_optim = torch.optim.Adam(params=self.actor_eval.parameters(), lr=1e-2)
        self.critic_optim = torch.optim.Adam(params=self.critic_eval.parameters(), lr=1e-2)
        self._buffer = np.array([]) #经验回放池
        self._memory_capacity = 1000 #经验池大小
        self._data_count = 0 #产生的数据总计数
        self._batch_size = 500 #从经验池中采样数量
        self._gamma = 0.99 #未来奖励的衰减系数

        #用actor估计的网络参数初始化actor目标的网络参数
        self.actor_target.load_state_dict(state_dict=self.actor_eval.state_dict())
        #用critic估计的网络参数初始化critic目标的网络参数
        self.critic_target.load_state_dict(state_dict=self.critic_eval.state_dict())

    def act(self, s1, s2):
        """
        产生动作
        :param s1:
        :param s2:
        :return:
        """
        s1 = torch.tensor(s1, dtype=torch.float64) #s1需要是二维矩阵
        s2 = torch.tensor(s2, dtype=torch.float64) #s2需要是二维矩阵
        a = self.actor_eval(s1, s2).detach().numpy()
        return a

    def store_transition(self, s, a, r, s_):
        """

        :param s: 是s1和s2拼接而成，s2需要先进行flatten
        :param a:
        :param r:
        :param s_: 同s
        :return:
        """
        trainsition = np.hstack((s, a, [r], s_))
        index = self._data_count % self._memory_capacity
        self._buffer[index, :] = trainsition
        self._data_count += 1

    def learn(self):
        #从经验回放池中抽样一个批次数据量
        samples = np.random.choice(a=self._memory_capacity, size=self._batch_size)
        batch_data = self._buffer[samples, :]
        s_index_limit = self._s1_dim+self._s2_dim[0]*self._s2_dim[1]
        s = torch.tensor(data=batch_data[:, :s_index_limit], dtype=torch.float64)
        a = torch.tensor(data=batch_data[:, s_index_limit:s_index_limit+self._a_dim], dtype=torch.float64)
        r = torch.tensor(data=batch_data[:, s_index_limit+self._a_dim:s_index_limit+self._a_dim+1], dtype=torch.float64)
        s_ = torch.tensor(data=batch_data[:, -s_index_limit:], dtype=torch.float64)

        def actor_learn():
            """
            演员训练更新
            :return:
            """
            a = self.actor_eval(s) #改s维度
            q_s_a = self.critic_eval(s, a)
            loss_a = -torch.mean(q_s_a)
            self.actor_optim.zero_grad()
            loss_a.backward()
            self.actor_optim.step()

        def critic_learn():
            """
            评论家训练更新
            :return:
            """
            a1 = self.actor_target(s_) #改s维度
            y_true = r + self._gamma * self.critic_target(s_, a1).detach().numpy()
            y_pred = self.critic_eval(s, a)
            loss_c_crition = torch.nn.MSELoss()
            loss_c = loss_c_crition(input=y_pred, target=y_true)
            self.critic_optim.zero_grad()
            loss_c.backward()
            self.critic_optim.step()

        def soft_update(net_target:torch.nn.Module, net_eval:torch.nn.Module, delta:float):
            """
            软更新
            :return:
            """
            for target_param, eval_param in zip(net_target.parameters(), net_eval.parameters()):
                target_param.data.copy_(target_param.data * (1.-delta) + eval_param.data * delta)

if __name__ == '__main__':
    a = np.arange(20).reshape(4, 5)
    b = np.random.choice(a=a.shape[-1], size=3)
    print(torch.tensor(a[:, b], dtype=torch.float64))
