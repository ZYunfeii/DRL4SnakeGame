#!/usr/bin/python
# -*- coding: utf-8 -*-
from Agent import AgentD3QN
from core import ReplayBuffer
from draw import Painter
from env4Snake import Snake
import random
import pygame
import numpy as np
import torch

def testAgent(agent):
    env = Snake()
    o = env.reset()
    for _ in range(500):
        env.render()
        for event in pygame.event.get():  # 不加这句render要卡，不清楚原因
            pass
        a = agent.select_action(o)
        o2, r, d = env.step(a)
        o = o2
        if d: break

if __name__ == "__main__":
    env = Snake()
    obs_dim = 3
    act_dim = 4
    agent = AgentD3QN()
    agent.init(256,obs_dim,act_dim)
    buffer = ReplayBuffer(2**15,obs_dim,1,if_on_policy=False,if_gpu=True)  # 离散情况这个buffer 的actdim一定写1（反直觉）
    MAX_EPISODE = 100
    MAX_STEP = 200
    batch_size = 256
    gamma = 0.99
    update_every = 50
    rewardList = []
    maxReward = -np.inf
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 10:
                a = agent.select_action(o)
            else:
                a = random.randint(0, 3)
            o2, r, d = env.step(a)
            mask = 0.0 if d else gamma
            buffer.append_buffer(o,(r,mask,a))

            if episode >= 10 and j % update_every == 0:
                agent.update_net(buffer,2**10,batch_size,1)
            o = o2
            ep_reward += r
            if d: break
        print('Episode:', episode, 'Reward:%f' %ep_reward)
        rewardList.append(ep_reward)

        if ep_reward > maxReward:
            maxReward = ep_reward
            print('已保存模型权重！')
            torch.save(agent.act.state_dict(),'act_weight.pkl')
        if episode > 10 and episode %2 == 0: testAgent(agent)

    painter = Painter(load_csv=False, load_dir=None)
    painter.addData(rewardList, 'D3QN')
    painter.drawFigure()


