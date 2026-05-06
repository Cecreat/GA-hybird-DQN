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
    def __init__(self, weights_1d, input_dim=6, hidden_dim1=32, hidden_dim2=32, output_dim=3):
        weights_1d = np.array(weights_1d)
        # 计算每一层权重的切片索引
        idx1 = input_dim * hidden_dim1
        idx2 = idx1 + hidden_dim1
        idx3 = idx2 + hidden_dim1 * hidden_dim2
        idx4 = idx3 + hidden_dim2
        idx5 = idx4 + hidden_dim2 * output_dim

        # 提取并重塑第一隐藏层 (Layer 1) 的 W 和 b
        self.W1 = weights_1d[0:idx1].reshape((input_dim, hidden_dim1))
        self.b1 = weights_1d[idx1:idx2]

        # 提取并重塑第二隐藏层 (Layer 2) 的 W 和 b
        self.W2 = weights_1d[idx2:idx3].reshape((hidden_dim1, hidden_dim2))
        self.b2 = weights_1d[idx3:idx4]

        # 提取并重塑输出层 (Output Layer) 的 W 和 b
        self.W3 = weights_1d[idx4:idx5].reshape((hidden_dim2, output_dim))
        self.b3 = weights_1d[idx5:]

    def forward(self, state):
        # 第一层：矩阵乘法 + ReLU 激活函数 (防止负值)
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1)

        # 第二层：矩阵乘法 + ReLU 激活函数
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)

        # 输出层：矩阵乘法 (不需要激活函数)
        z3 = np.dot(a2, self.W3) + self.b3

        # 返回 Q 值最大的那个动作的索引 (0, 1 或 2)
        return np.argmax(z3)
#定义我们的目标函数
def Objective(decoded_weights, render=False):
    env = SimulationEnv()
    mlp = NumpyMLP(decoded_weights)
    state = env.reset()

    initial_dist = env.agent.pos.distance_to(env.target_pos)
    # 【新增】记录整个过程中的最小距离（防止它半路往回跑刷分）
    min_dist = initial_dist
    survival_step = 0
    done = False

    # pygame循环
    while not done and survival_step < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = mlp.forward(state)
        real_action = action - 1
        state, reward, done = env.step(real_action)
        survival_step += 1

        # 【新增】实时更新历史离目标最近的距离
        current_dist = env.agent.pos.distance_to(env.target_pos)
        if current_dist < min_dist:
            min_dist = current_dist

        if render:
            env.render()

    # ==========================================
    # 【核心重构】在循环结束后计算最终适应度得分
    # ==========================================
    final_dist = env.agent.pos.distance_to(env.target_pos)

    # 使用历史最近距离，哪怕它最后被撞死了，只要它曾经很接近终点，也算它牛！
    distance_progress = initial_dist - min_dist

    # 1. 基础分：大幅提高距离权重，大幅降低生存步数权重
    score = distance_progress * 1.0 + survival_step * 0.1

    # 2. 死亡惩罚 (没到终点且中途撞死)
    if done and final_dist > 20 and survival_step < 500:
        score *= 0.5  # 撞墙直接成绩打半折

    # 3. 胜利超级大奖与效率奖励 (到达终点)
    if final_dist <= 20:
        # 到达奖励 1000分！并且用时越少，奖励越高 (500-survival_step)
        score += 1000 + (500 - survival_step) * 2.0

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
    num_params = 1379  # 6*12 + 12 + 12*3 + 3
    boundary = [[-1.0, 1.0]] * num_params  # 所有权重限制在 -1 到 1 之间
    n_bits = 16  # 16位精度
    n_pop = 100  # 种群规模
    n_it = 50  # 迭代代数
    cross_rate = 0.9
    # 变异率
    mutation_rate = 1.0 / (float(n_bits * len(boundary)))

    print(f"开始进化！每条染色体长度为 {n_bits * num_params} 位...")
    best_chromosome, final_score = Genetic_Algorithm(boundary, n_bits, n_it, n_pop, cross_rate, mutation_rate)

    print(f"\n进化完成！最高适应度得分为: {final_score:.2f}")

    # 将最优二进制染色体解码，并保存为 NumPy 数组
    best_weights = decoder(boundary, n_bits, best_chromosome)
    np.save("../best_ga_weights.npy", np.array(best_weights))
    print("最优个体权重已保存至 best_ga_weights.npy")

    print("准备全屏渲染展示最优个体的生存本能...")
    Objective(best_weights, render=True)




