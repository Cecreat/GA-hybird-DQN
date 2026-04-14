import sys

import numpy as np
import random
from numpy.random import randint
from numpy.random import rand
import math
import pygame
from tensorflow_probability.python.internal.backend.jax import reshape

from Env import SimulationEnv


# 定义6x12x3的神经网络
class NumpyMLP:
    def __init__(self,flat_weights):
        self.w1 =np.array(flat_weights[0:72]).reshape((6,12))
        self.b1 = np.array(flat_weights[72:84])
        self.w2 = np.array(flat_weights[84:120]).reshape((12,3))
        self.b2 = np.array(flat_weights[120:123])
    # 神经网络的前向推理逻辑
    def forward(self,state):
        x=np.array(state)#获取状态值并转化为np数组
        z1=np.dot(x,self.w1)+self.b1#获得输入后，将输入矩阵和权重矩阵相乘后再加上偏置值
        a1=np.maximum(0,z1)#通过激活函数ReLU
        z2=np.dot(a1,self.w2)+self.b2#计算输出矩阵
        return np.argmax(z2)#从输出的离散动作空间中找出神经网络计算的Q值最大的那一个。注意这里我们求的是最大值的下标是因为神经网络计算的是Q值，所以最终输出的离散动作的范围为0-2，后期要映射为-1~1

#定义我们的目标函数
def Objective(decoded_weights, render=False):
    #因为我们使用的是pygame环境进行模拟，所以先要实例化环境并且获得MLP的推理结果 然后reset环境获取最新state
    env = SimulationEnv()
    mlp = NumpyMLP(decoded_weights)
    state = env.reset()

    #适应度得分计算是基于移动距离以及agent的生存步共同计算
    #初始化与目标的距离、生存步以及存活标记
    initial_dist=env.agent.pos.distance_to(env.target.pos)
    survival_step=0
    done=False

    #pygame循环
    while not done and survival_step<500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    # 推理新的动作
    action=mlp.forward(state)
    # 映射为环境的离散动作空间
    real_action=action-1
    # 获取环境步进后的新的状态值、奖励值、存活标记
    state,reward,done,info=env.step(real_action)
    # 增加生存步
    survival_step+=1

    if render:
        env.render()


    # 步进后的与目标距离
    final_dist=env.agent.pos.distance_to(env.target.pos)
    # agent移动的距离
    distance_progress=initial_dist-final_dist
    # 计算适应度得分
    score=survival_step*1.0+distance_progress*0.5

    # 如果死亡、最终离目标距离大于20 并且生存步小于500那么得分打半折
    if done and final_dist>20 and survival_step<500:
        score *=0.5

    return score

# 编写解码器，本质是对2进制基因片段进行切片
def decoder(boundary, n_bits, bitstring):
    decoded=[]
    largest=2**n_bits-1
    for i in range(len(boundary)):
        start, end= i*n_bits, (i+1)*n_bits
        substring = bitstring[start:end]
        #为了将二进制基因片段解码为十进制实际神经网络的参数值，我们需要先将二进制数组转化为字符串
        chars=''.join([str(s) for s in substring])
        integer= int(chars,2)
        # 解码的值再映射到搜索空间内，映射方式为搜索空间的下界加上计算出的值在搜索空间内占的比例
        value= boundary[i][0]+(integer/largest)*(boundary[i][1]-boundary[i][0])
        decoded.append(value)
    return decoded


