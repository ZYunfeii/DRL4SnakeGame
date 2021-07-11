# DRL4SnakeGame

 Using deep reinforcement learning to play Snake game(贪吃蛇).

The used algorithm is PPO for discrete! It has the brilliant performance in the field of discrete action space just like in continuous action space.

You just need half an hour to train the snake and then it can be as smart as you.

## Result

<img src="README.assets\Figure_1-1625922043908.png" width="300" height="200" alt="Figure_1" style="zoom: 50%;" /><img src="README.assets\result.gif" width="300" height="200" alt="result" style="zoom:50%;" />

<img src="README.assets\Figure_1-1625980553553.png" width="300" height="200" alt="Figure_1" style="zoom:50%;" /><img src="README.assets\result-1625980590643.gif" width="300" height="200" alt="result" style="zoom:50%;" />

## File to Illustrate

Agent.py: file to store the algorithm.

core.py: file to store the net for DRL algorithm.

draw.py: file to draw the reward curve.

env4Snake.py: the environment for snake game.

main.py: the main func.

**what you need to do is to run the main.py and then run the env4Snake to test your mdoel!**

## Requirements

1. torch
2. numpy
3. seaborn==0.11.1
4. pygame==2.0.1
5. matplotlib==3.2.2
6. PIL(This is unimportant. I do not use this package in running process.)

## Details for realization

See my blog for details: https://blog.csdn.net/weixin_43145941/article/details/118639211

