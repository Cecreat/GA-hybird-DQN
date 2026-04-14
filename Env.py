import pygame
import math
import random
from pygame.math import Vector2

# --- 环境超参数 ---
WIDTH, HEIGHT = 800, 600
FPS = 60
MAX_RAY_DIST = 200  # 射线最大探测距离
NUM_RAYS = 5  # 5 根射线
FOV = 90  # 90度视野


class DynamicObstacle:
    def __init__(self):
        # 设置了障碍物的的初始出生位置，随机出生在中间地带
        self.pos = Vector2(random.randint(100, 700), random.randint(100, 500))
        # 定义了障碍物的物理和渲染半径，后续智能体的射线探测和刚体碰撞检测都依赖于这个半径
        self.radius = 20
        # 随机移动方向和速度 random.choice([-1, 1])在x和y轴随机挑选-1和1.产生四个反向（左上、左下、右上、右下）并进行归一化将方向缩放为单位向量确保斜向速度。最后乘上速度计算出速度向量
        self.velocity = Vector2(random.choice([-1, 1]), random.choice([-1, 1])).normalize() * 2

    # 定义每一帧障碍物的如何进行运动
    def update(self):
        # 更新位置：位置向量和速度向量相加实现位移
        self.pos += self.velocity
        # 简单的边界反弹逻辑如果触碰到边界就将对应方向的速度取反进行反方向移动
        if self.pos.x < 0 or self.pos.x > WIDTH: self.velocity.x *= -1
        if self.pos.y < 0 or self.pos.y > HEIGHT: self.velocity.y *= -1
    # 屏幕渲染逻辑
    def draw(self, surface):
        pygame.draw.circle(surface, (200, 50, 50), (int(self.pos.x), int(self.pos.y)), self.radius)

# 智能体类
class Agent:
    def __init__(self):
        # 生成的智能体实例的初始世界坐标，但是SimulationEnv类的reset方法会在每回合开始的时候覆盖这个值，将智能体的放置在随机位置，以防止模型对特定的出生点产生过拟合
        self.pos = Vector2(100, 100)
        # 智能体的碰撞体积和渲染半径，小于障碍物的20是为了增加穿梭缝隙的容错率
        self.radius = 15
        self.heading = 0.0  # 偏航角 (度)
        # 智能体的初始速度，意味着在没有碰撞的情况下，智能体会沿着当前方向heading恒定移动3个像素
        self.speed = 3.0

    # 智能体移动原理，action_steer为神经网络传入的离散动作指令
    def move(self, action_steer):
        # action_steer: 转向量 (例如 -1 左转, 0 保持, 1 右转)
        # 偏航角根据传入的离散动作指令修改，例如（传入-1左转，那么heading就在原有数值上加上-1*5，代表向左（逆时针）转5度）
        self.heading += action_steer * 5.0

        # 局部坐标到世界坐标的转换机制
        # 通过将世界坐标正方向的向量按照智能体的航向角进行旋转从而得到在世界坐标系下的智能体的坐标
        forward = Vector2(1, 0).rotate(self.heading)
        # 按照该世界坐标位置向量乘以速度向量和当前位置向量相加得到移动后的位置
        self.pos += forward * self.speed

    def draw(self, surface):
        pygame.draw.circle(surface, (50, 150, 250), (int(self.pos.x), int(self.pos.y)), self.radius)
        # 绘制朝向指示线
        forward = Vector2(1, 0).rotate(self.heading)
        #定义朝向指向线的终点：以智能体的圆心位置延伸距离为世界坐标向量乘以渲染半径的距离
        end_pos = self.pos + forward * self.radius
        # 渲染
        pygame.draw.line(surface, (0, 255, 0), self.pos, end_pos, 2)


class SimulationEnv:
    def __init__(self):
        # 游戏初始化
        pygame.init()
        # 渲染游戏窗口
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("GA-RL Autonomous Navigation Env")
        # 将物理更新和画面渲染严格锁定在60FPS
        self.clock = pygame.time.Clock()

        # 提前声明环境中的三大实体
        self.agent = Agent()
        self.obstacles = []
        self.target_pos = Vector2()

    # 自动回合重置功能
    def reset(self):
        # 重置智能体到随机位置防止过拟合
        self.agent = Agent()
        self.agent.pos = Vector2(random.randint(50, 150), random.randint(50, 550))
        self.agent.heading = random.uniform(0, 360)

        # 随机刷新目标航点，固定刷新在屏幕右半边，迫使智能体必须穿越中间智能体地带
        self.target_pos = Vector2(random.randint(650, 750), random.randint(50, 550))

        # 初始化多源移动障碍物
        self.obstacles = [DynamicObstacle() for _ in range(10)]

        # 重置完毕后立即调用感知函数获取第一帧的观察状态，并返回给算法，作为网络推理的初始输入
        return self._get_state()

    # 定义感知射线
    def _cast_rays(self):
        # 空列表用来存放5根射线的探测距离。
        ray_distances = []
        # 定义了五根射线的起始相对（智能体）角度为-45，FOV代表智能体的视野为90
        start_angle = -FOV / 2
        # 五根射线，每一根射线之间的角度应该为FOV被均分为4份
        angle_step = FOV / (NUM_RAYS - 1)

        # 遍历射线,计算他们的绝对角度和生成对应方向的单位向量
        for i in range(NUM_RAYS):
            # 射线绝对角度=智能体的航向角+起始射线的相对角度+当前射线相对于起始射线的角度步
            ray_angle = self.agent.heading + start_angle + (i * angle_step)
            #射线对应方向单位向量=将世界正方向单位向量旋转其绝对角度
            ray_dir = Vector2(1, 0).rotate(ray_angle)

            # 射线步进检测 (简化版的连续射线投射)
            # 定义射线的预设初始状态
            distance = MAX_RAY_DIST #探测到的最远距离,200像素
            hit = False #使否击中任何物体
            # 步进射线检测,
            for step in range(1, MAX_RAY_DIST, 5):  # 步长为 5 像素
                # 设置虚拟测试点,沿着ray_dir方向,每5个像素设置一个测试点,步长代表我们探测的距离单位是5像素
                test_point = self.agent.pos + ray_dir * step

                # 检查当前测试点与所有动态障碍物的碰撞
                for obs in self.obstacles:
                    if test_point.distance_to(obs.pos) <= obs.radius:#如果测试点距离障碍物的距离小于等于障碍物的渲染距离
                        # 探测到的距离就为当前步长
                        distance = step
                        # 将击中标志置为True
                        hit = True
                        # 跳出循环
                        break

                # 检查边界碰撞
                if not (0 <= test_point.x <= WIDTH and 0 <= test_point.y <= HEIGHT):#如果测试点的x坐标不大于等于0,小于等于窗口的宽度 并且y坐标不大于等于0,小于等于高度
                    # 那么探测到的距离就为步长
                    distance = step
                    #  将击中标志置为True
                    hit = True

                if hit:
                    break

            # 距离衰减数据归一化处理 (0 到 1)
            normalized_dist = distance / MAX_RAY_DIST#当前探测到的距离占最大探测距离的多少
            ray_distances.append(normalized_dist)

            # 渲染射线 (仅用于可视化调试)
            end_point = self.agent.pos + ray_dir * distance
            pygame.draw.line(self.screen, (0, 255, 0), self.agent.pos, end_point, 1)
        # 返回所有射线探测到的距离
        return ray_distances

    def _get_state(self):
        #获取射线探测到的5维距离数据
        rays = self._cast_rays()

        # 计算相对目标偏航角 (theta_target)
        # 计算与目标点之间的横坐标距离和纵坐标距离
        dir_to_target = self.target_pos - self.agent.pos
        # 目标角度等于两个坐标距离的反正切值
        target_angle = math.degrees(math.atan2(dir_to_target.y, dir_to_target.x))
        # 计算角度差(通过数学公式计算每次最近的转弯方向)并归一化到 [-1, 1]
        angle_diff = (target_angle - self.agent.heading + 180) % 360 - 180
        normalized_yaw = angle_diff / 180.0

        # 返回 6 维状态向量: [ray1-distance, ray2-distance, ray3-distance, ray4-distance, ray5-distance, normalized_yaw]
        state = rays + [normalized_yaw]
        return state

    # 将每一帧所有需要执行的方法打包到一个方法中，并将最新状态和奖励以及存活标志一起返回 方法包括：智能体的移动、障碍物的移动、存活和奖励机制
    def step(self, action):
        """执行单步动作，更新环境并返回下一个状态、奖励和是否结束标志"""
        self.agent.move(action)

        for obs in self.obstacles:
            obs.update()

        # --- 刚体碰撞检测机制  ---

        done = False#存活标志
        reward = 0.1  # 存活奖励

        # 检测障碍物碰撞 (失败条件)
        for obs in self.obstacles:
            if self.agent.pos.distance_to(obs.pos) < (self.agent.radius + obs.radius):
                done = True
                reward = -10.0  # 碰撞惩罚

        # 边界检测
        if self.agent.pos.x < 0 or self.agent.pos.x > WIDTH or self.agent.pos.y < 0 or self.agent.pos.y > HEIGHT:
            done = True
            reward = -10.0

        # 检测目标到达 (胜利条件)
        if self.agent.pos.distance_to(self.target_pos) < 20:
            done = True
            reward = 20.0

        next_state = self._get_state()
        print(f"最新状态为：{next_state}，当前奖励值：{reward}，是否死亡{done}")
        return next_state, reward, done

    # 渲染所有实体和背景以及控制帧率
    def render(self):
        """支持 60FPS 渲染 """
        self.screen.fill((30, 30, 30))  # 深色背景

        # 绘制目标点
        pygame.draw.circle(self.screen, (50, 200, 50), (int(self.target_pos.x), int(self.target_pos.y)), 15)

        for obs in self.obstacles:
            obs.draw(self.screen)

        self.agent.draw(self.screen)

        # 这里必须在绘制完实体后再次调用_cast_rays来渲染射线可视化
        self._cast_rays()

        # 将绘制好的实体推送到显示器上
        pygame.display.flip()
        self.clock.tick(FPS)  # 控制帧率


# --- 测试运行循环 ---
if __name__ == "__main__":
    env = SimulationEnv()
    state = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 随机动作测试：-1左转，0直行，1右转
        action = random.choice([-1, 0, 1])
        next_state, reward, done = env.step(action)
        if done==True:
           print("------------------------------------------------------------------------------------------------")
        env.render()
        pygame.display.flip()
        env.render()

        if done:
            env.reset()

    pygame.quit()