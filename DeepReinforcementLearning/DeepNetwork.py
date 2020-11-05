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
import copy

class Actor(torch.nn.Module):
    def __init__(self, s1_dim, s2_dim, a_dim):
        """

        :param s1_dim: 8
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

    def forward(self, input:torch.Tensor):
        """

        :param input: torch.Tensor, shape=(batch_size, 136)
        :return:
        """
        s1, s2 = torch.split(tensor=copy.deepcopy(input), split_size_or_sections=[8, 128], dim=1) #输入维度为(batch_size, 8),(batch_size, 128)
        s1net = self._s1linear1(s1) #shape=(batch_size, 28)
        s1net = F.relu(input=s1net)
        s2net = self._s2linear1(s2) #shape=(batch_size, 100)
        s2net = F.relu(input=s2net)
        s2net = s2net.reshape(shape=[-1, 1, 10, 10]) #shape=(batch_size, 1, 10, 10)
        s2net = self._s2net_sequential1(s2net) #shape=(batch_size, 1, 10, 10)
        s2net = s2net.reshape(shape=[-1, reduce(lambda x, y:x*y, s2net.size()[1:])]) #shape=(batch_size, 100)
        a = self._combinelinear(torch.cat(tensors=[s1net, s2net], dim=1)) #shape=(batch_size, 64)
        a = F.tanh(input=a)
        return a

class Critic(torch.nn.Module):
    def __init__(self, s1_dim, s2_dim, a_dim):
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
        self._anet = torch.nn.Sequential(torch.nn.Linear(a_dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128))
        self._combinelinear = torch.nn.Linear(in_features=128+128, out_features=1)

    def forward(self, input:torch.Tensor):
        """

        :param input: s1, s2, a
        :return:
        """
        s1, s2, a = torch.split(tensor=copy.deepcopy(input), split_size_or_sections=[8, 128, 64], dim=1) # 输入维度为(batch_size, 8),(batch_size, 128),(batch_size, 64)
        s1net = self._s1linear1(s1) #shape=(batch_size, 28)
        s1net = F.relu(input=s1net)
        s2net = self._s2linear1(s2) #shape=(batch_size, 100)
        s2net = F.relu(input=s2net)
        s2net = s2net.reshape(shape=[-1, 1, 10, 10]) #shape=(batch_size, 1, 10, 10)
        s2net = self._s2net_sequential1(s2net) #shape=(batch_size, 1, 10, 10)
        s2net = s2net.reshape(shape=[-1, reduce(lambda x, y: x * y, s2net.size()[1:])]) #shape=(batch_size, 100)
        anet = self._anet(a) #shape=(batch_size, 128)
        q = self._combinelinear(torch.cat(tensors=[s1net, s2net, anet], dim=1)) #shape=(batch_size, 1)
        q = torch.tanh(q)
        return q

class Agent:
    def __init__(self):
        self._s1_dim = 8
        self._s2_dim = 128
        self._a_dim = 64
        self.actor_eval = Actor(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.actor_target = Actor(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.critic_eval = Critic(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.critic_target = Critic(s1_dim=self._s1_dim, s2_dim=self._s2_dim, a_dim=self._a_dim)
        self.actor_optim = torch.optim.Adam(params=self.actor_eval.parameters(), lr=1e-3) #1e-3最好
        self.critic_optim = torch.optim.Adam(params=self.critic_eval.parameters(), lr=1e-3) #1e-3最好
        self._buffer = np.array([]) #经验回放池
        self._memory_capacity = 2000 #1000 #经验池大小
        self._data_count = 0 #产生的数据总计数
        self._batch_size = 1000 #1000最好 #从经验池中采样数量
        self._gamma = 0.99 #未来奖励的衰减系数

        #用actor估计的网络参数初始化actor目标的网络参数
        self.actor_target.load_state_dict(state_dict=self.actor_eval.state_dict())
        #用critic估计的网络参数初始化critic目标的网络参数
        self.critic_target.load_state_dict(state_dict=self.critic_eval.state_dict())

    def act(self, s):
        """
        产生动作
        :param s:
        :return:
        """
        s = torch.tensor(data=s, dtype=torch.float32)
        a = self.actor_eval(s).detach().numpy()
        return a

    def store_transition(self, s, a, r, s_):
        """

        :param s: 是s1和s2拼接而成，s2需要先进行flatten
        :param a:
        :param r:
        :param s_: 同s
        :return:
        """
        trainsition = np.hstack((s, a, np.array([r]).reshape(1, -1), s_))
        if self._buffer.shape[0] < self._memory_capacity:
            # print(self._buffer.shape)
            self._buffer = np.vstack((self._buffer, trainsition)) if self._buffer.shape[0] else trainsition
        else:
            index = self._data_count % self._memory_capacity
            self._buffer[index, :] = trainsition

        self._data_count += 1

    def learn(self):
        if self._buffer.shape[0] < self._memory_capacity:
            return
        #从经验回放池中抽样一个批次数据量
        samples = np.random.choice(a=self._memory_capacity, size=self._batch_size)
        batch_data = self._buffer[samples, :]
        s_index_limit = self._s1_dim+self._s2_dim
        s = torch.tensor(data=batch_data[:, :s_index_limit], dtype=torch.float32)
        a = torch.tensor(data=batch_data[:, s_index_limit:s_index_limit+self._a_dim], dtype=torch.float32)
        r = torch.tensor(data=batch_data[:, s_index_limit+self._a_dim:s_index_limit+self._a_dim+1], dtype=torch.float32)
        s_ = torch.tensor(data=batch_data[:, -s_index_limit:], dtype=torch.float32)

        def actor_learn():
            """
            演员训练更新
            :return:
            """
            a = self.actor_eval(s)
            q_s_a = self.critic_eval(torch.cat(tensors=[s, a], dim=1))
            loss_a = -torch.mean(q_s_a)
            self.actor_optim.zero_grad()
            loss_a.backward()
            self.actor_optim.step()

        def critic_learn():
            """
            评论家训练更新
            :return:
            """
            a1 = self.actor_target(s_)
            y_true = r + self._gamma * self.critic_target(torch.cat(tensors=[s_, a1], dim=1)).detach().numpy()
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

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic_eval, 0.02)
        soft_update(self.actor_target, self.actor_eval, 0.02)

if __name__ == '__main__':
    a = torch.nn.Conv2d(1, 1, 2)
    print(a.parameters().__next__().__class__, a.parameters().__next__().data.__class__)
