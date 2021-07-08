#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import pygame
import sys
from pygame.locals import *
import numpy as np
from Agent import AgentD3QN
import torch

class Snake:
    def __init__(self):
        self.snake_speed = 15 # 贪吃蛇的速度
        self.windows_width = 800
        self.windows_height = 600  # 游戏窗口的大小
        self.cell_size = 20  # 贪吃蛇身体方块大小,注意身体大小必须能被窗口长宽整除
        self.map_width = int(self.windows_width / self.cell_size)
        self.map_height = int(self.windows_height / self.cell_size)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (230, 230, 230)
        self.dark_gray = (40, 40, 40)
        self.DARKGreen = (0, 155, 0)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.dark_blue = (0, 0, 139)
        self.BG_COLOR = self.black  # 游戏背景颜色
        # 定义方向
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3

        self.HEAD = 0  # 贪吃蛇头部下标

        pygame.init()  # 模块初始化

        [self.snake_coords,self.direction,self.food,self.state] = [None,None,None,None]

    def reset(self):
        self.snake_speed_clock = pygame.time.Clock()  # 创建Pygame时钟对象
        self.screen = pygame.display.set_mode((self.windows_width, self.windows_height))
        self.screen.fill(self.white)
        startx = random.randint(3, self.map_width - 8)  # 开始位置
        starty = random.randint(3, self.map_height - 8)
        self.snake_coords = [{'x': startx, 'y': starty},  # 初始贪吃蛇
                        {'x': startx - 1, 'y': starty},
                        {'x': startx - 2, 'y': starty}]
        self.direction = self.RIGHT  # 开始时向右移动
        self.food = self.get_random_location()  # 实物随机位置
        return self.getState()

    def step(self,action):
        if action == self.LEFT and self.direction != self.RIGHT:
            self.direction = self.LEFT
        elif action == self.RIGHT and self.direction != self.LEFT:
            self.direction = self.RIGHT
        elif action == self.UP and self.direction != self.DOWN:
            self.direction = self.UP
        elif action == self.DOWN and self.direction != self.UP:
            self.direction = self.DOWN
        self.move_snake(self.direction,self.snake_coords)
        ret = self.snake_is_alive(self.snake_coords)
        d = True if not ret else False
        flag = self.snake_is_eat_food(self.snake_coords, self.food)
        reward = self.getReward(flag,d)

        return [self.getState(),reward,d]

    def getReward(self,flag, d):
        reward = 0
        if flag:
            reward += 500
        [xhead,yhead] = [self.snake_coords[self.HEAD]['x'],self.snake_coords[self.HEAD]['y']]
        [xfood,yfood] = [self.food['x'],self.food['y']]
        distance1 = np.sqrt((xhead-xfood)**2+(yhead-yfood)**2)
        if distance1 < 5:
            reward += (5-distance1)/5
        if d: reward -= 50
        return reward


    def render(self):
        self.screen.fill(self.BG_COLOR)
        self.draw_snake(self.screen,self.snake_coords)
        self.draw_food(self.screen,self.food)
        self.draw_score(self.screen,len(self.snake_coords)-3)
        pygame.display.update()
        self.snake_speed_clock.tick(self.snake_speed) #控制fps



    def getState(self):
        [xhead, yhead] = [self.snake_coords[self.HEAD]['x'], self.snake_coords[self.HEAD]['y']]
        [xfood, yfood] = [self.food['x'], self.food['y']]
        deltax = (xfood - xhead)/self.map_width
        deltay = (yfood - yhead)/self.map_height
        # xhead/self.map_height,yhead/self.map_width
        return [deltax,deltay,self.direction/3]


    # 将食物画出来
    def draw_food(self,screen, food):
        x = food['x'] * self.cell_size
        y = food['y'] * self.cell_size
        appleRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, self.Red, appleRect)

    # 将贪吃蛇画出来
    def draw_snake(self,screen, snake_coords):
        for coord in snake_coords:
            x = coord['x'] * self.cell_size
            y = coord['y'] * self.cell_size
            wormSegmentRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, self.dark_blue, wormSegmentRect)
            wormInnerSegmentRect = pygame.Rect(  # 蛇身子里面的第二层亮绿色
                x + 4, y + 4, self.cell_size - 8, self.cell_size - 8)
            pygame.draw.rect(screen, self.blue, wormInnerSegmentRect)

    # 移动贪吃蛇
    def move_snake(self,direction, snake_coords):
        if direction == self.UP:
            newHead = {'x': snake_coords[self.HEAD]['x'], 'y': snake_coords[self.HEAD]['y'] - 1}
        elif direction == self.DOWN:
            newHead = {'x': snake_coords[self.HEAD]['x'], 'y': snake_coords[self.HEAD]['y'] + 1}
        elif direction == self.LEFT:
            newHead = {'x': snake_coords[self.HEAD]['x'] - 1, 'y': snake_coords[self.HEAD]['y']}
        elif direction == self.RIGHT:
            newHead = {'x': snake_coords[self.HEAD]['x'] + 1, 'y': snake_coords[self.HEAD]['y']}
        else:
            print('error')

        snake_coords.insert(0, newHead)

    # 判断蛇死了没
    def snake_is_alive(self,snake_coords):
        tag = True
        if snake_coords[self.HEAD]['x'] == -1 or snake_coords[self.HEAD]['x'] == self.map_width or snake_coords[self.HEAD]['y'] == -1 or \
                snake_coords[self.HEAD]['y'] == self.map_height:
            tag = False  # 蛇碰壁啦
        for snake_body in snake_coords[1:]:
            if snake_body['x'] == snake_coords[self.HEAD]['x'] and snake_body['y'] == snake_coords[self.HEAD]['y']:
                tag = False  # 蛇碰到自己身体啦
        return tag

    # 判断贪吃蛇是否吃到食物
    def snake_is_eat_food(self,snake_coords, food):  # 如果是列表或字典，那么函数内修改参数内容，就会影响到函数体外的对象。
        flag = False
        if snake_coords[self.HEAD]['x'] == food['x'] and snake_coords[self.HEAD]['y'] == food['y']:
            food['x'] = random.randint(0, self.map_width - 1)
            food['y'] = random.randint(0, self.map_height - 1)  # 实物位置重新设置
            flag = True
        else:
            del snake_coords[-1]  # 如果没有吃到实物, 就向前移动, 那么尾部一格删掉
        return flag

    # 食物随机生成
    def get_random_location(self):
        return {'x': random.randint(0, self.map_width - 1), 'y': random.randint(0, self.map_height - 1)}


    # 画成绩
    def draw_score(self,screen, score):
        font = pygame.font.Font('myfont.ttf', 30)
        scoreSurf = font.render('得分: %s' % score, True, self.Green)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (self.windows_width - 120, 10)
        screen.blit(scoreSurf, scoreRect)

    @staticmethod
    # 程序终止
    def terminate():
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    env = Snake()
    agent = AgentD3QN()
    agent.init(256, 3, 4)
    agent.act.load_state_dict(torch.load('act_weight.pkl'))
    for _ in range(15):
        o = env.reset()
        while True:
            env.render()
            for event in pygame.event.get(): # 不加这句render要卡，不清楚原因
                pass
            a = agent.select_action(o)
            o2,r,d = env.step(a)
            o = o2
            if d: break


