#!/usr/bin/python
# -*- coding: utf-8 -*-
from Agent import AgentD3QN, AgentDiscretePPO
from core import ReplayBuffer
from draw import Painter
from env4Snake import Snake
import random
import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt

def testAgent(test_env,agent,episode):
    ep_reward = 0
    o = test_env.reset()
    for _ in range(500):
        if episode % 100 == 0:
            test_env.render()
        for event in pygame.event.get():  # 不加这句render要卡，不清楚原因
            pass
        a_int, a_prob = agent.select_action(o)
        o2, reward, done, _ = test_env.step(a_int)
        ep_reward += reward
        if done: break
        o = o2
    return ep_reward

if __name__ == "__main__":
    env = Snake()
    test_env = Snake()
    act_dim = 4
    obs_dim = 5
    agent = AgentDiscretePPO()
    agent.init(512,obs_dim,act_dim,if_use_gae=True)
    agent.state = env.reset()
    buffer = ReplayBuffer(2**15,obs_dim,act_dim,True)
    MAX_EPISODE = 500
    MAX_STEP = 200
    batch_size = 256
    update_every = 50
    rewardList = []
    maxReward = -np.inf

    for episode in range(MAX_EPISODE):
        with torch.no_grad():
            trajectory_list = agent.explore_env(env,2**12,1,0.99)
        buffer.extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer,batch_size,1,2**-8)
        ep_reward = testAgent(test_env, agent, episode)
        print('Episode:', episode, 'Reward:%f' % ep_reward)
        rewardList.append(ep_reward)
        if episode > MAX_EPISODE - 100 and ep_reward > maxReward:
            maxReward = ep_reward
            print('保存模型！')
            torch.save(agent.act.state_dict(),'act_weight.pkl')
    pygame.quit()

    painter = Painter(load_csv=False, load_dir=None)
    painter.addData(rewardList, 'PPO')
    painter.saveData('reward.csv')
    painter.drawFigure()


