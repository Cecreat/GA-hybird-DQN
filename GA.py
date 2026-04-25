import sys

import numpy as np
import random
from numpy.random import randint
from numpy.random import rand
import math
import pygame


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
def Objective(decoded_weights,render=False):
    #因为我们使用的是pygame环境进行模拟，所以先要实例化环境并且获得MLP的推理结果 然后reset环境获取最新state
    env = SimulationEnv()
    mlp = NumpyMLP(decoded_weights)
    state = env.reset()

    #适应度得分计算是基于移动距离以及agent的生存步共同计算
    #初始化与目标的距离、生存步以及存活标记
    initial_dist=env.agent.pos.distance_to(env.target_pos)
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
        state,reward,done=env.step(real_action)
        # 增加生存步
        survival_step+=1

        if render:
            env.render()


        # 步进后的与目标距离
        final_dist=env.agent.pos.distance_to(env.target_pos)
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
# 锦标赛逻辑
def selection(pop, scores, k=3):
    # 随机选择种群中的任意一个个体（下标）
    selected_ix=randint(len(pop))
    # 遍历整个种群，选择k-1个参赛个体
    for ix in randint(0,len(pop),k-1):
        # 如果当前个体的得分大于当前选中个体
        if scores[ix]>scores[selected_ix]:
            selected_ix=ix
    return pop[selected_ix]

# 基因交叉逻辑
def crossover(p1,p2,cross_rate):
    # 先复制父母的基因
    c1,c2=p1.copy(),p2.copy()
    # 随机数小于交叉率
    if rand()<cross_rate:
        # 选取交叉点位
        cp =randint(1,len(p1)-2)#防止末端导致的未变化
        c1=p1[:cp]+p2[cp:]
        c2=p2[:cp]+p1[cp:]
    return [c1,c2]

# 基因变异逻辑
def mutation(bitstring,mutation_rate):
    r_func= random.random
    for i in range (len(bitstring)):
        if r_func()<mutation_rate:
            bitstring[i]=1-bitstring[i]
    return bitstring

def Genetic_Algorithm(boundary, n_bits, n_it,n_pop, cross_rate, mutation_rate):
    total_bits=n_bits*len(boundary)
    pop=[randint(0,2,total_bits).tolist() for _ in range(n_pop)]

    best=pop[0]
    best_score=Objective(decoder(boundary, n_bits, pop[0]))

    for gen in range(n_it):
        print(f"------正在评估第{gen+1}/{n_it}代种群------")
        decoded=[decoder(boundary, n_bits, p)for p in pop]

        scores=[Objective(d)for d in decoded]

        for i in range (n_pop):
            if scores[i]>best_score:
                best,best_score=pop[i],scores[i]
                print(f">第{gen+1}代发现新的优胜者！得分:{best_score:.2f}")

        selected =[selection(pop,scores)for _ in range(n_pop)]
        children=list()
        for i in range(0,n_pop,2):
            p1,p2=selected[i],selected[i+1]
            for c in crossover(p1,p2,cross_rate):
                children.append(mutation(c,mutation_rate))
        pop = children

    return best, best_score



if __name__ == "__main__":
    # 参数配置
    num_params = 123  # 6*12 + 12 + 12*3 + 3
    boundary = [[-1.0, 1.0]] * num_params  # 所有权重限制在 -1 到 1 之间
    n_bits = 16  # 16位精度
    n_pop = 30  # 种群规模
    n_it = 30  # 迭代代数
    cross_rate = 0.9
    # 变异率
    mutation_rate = 1.0 / (float(n_bits * len(boundary)))

    print(f"开始进化！每条染色体长度为 {n_bits * num_params} 位...")
    best_chromosome, final_score = Genetic_Algorithm(boundary, n_bits, n_it, n_pop, cross_rate, mutation_rate)

    print(f"\n进化完成！最高适应度得分为: {final_score:.2f}")

    # 将最优二进制染色体解码，并保存为 NumPy 数组
    best_weights = decoder(boundary, n_bits, best_chromosome)
    np.save("best_ga_weights.npy", np.array(best_weights))
    print("最优个体权重已保存至 best_ga_weights.npy")

    print("准备全屏渲染展示最优个体的生存本能...")
    Objective(best_weights, render=True)




