import pygame
import math
import random
from pygame.math import Vector2
from Rays_Explore import *

# --- 环境超参数 ---
WIDTH, HEIGHT = 800, 600
FPS = 60
MAX_RAY_DIST = 200
NUM_RAYS = 11
FOV = 120


class DynamicObstacle:
    def __init__(self, rng=None):
        """
        动态障碍物。

        rng 用于固定随机种子评估。不要直接依赖全局 random，
        否则不同环境实例之间会互相影响随机序列。
        """
        self.rng = rng if rng is not None else random
        self.pos = Vector2(
            self.rng.randint(100, 700),
            self.rng.randint(100, 500)
        )
        self.radius = 18
        self.velocity = Vector2(
            self.rng.choice([-1, 1]),
            self.rng.choice([-1, 1])
        ).normalize() * 2

    def update(self):
        self.pos += self.velocity
        if self.pos.x - self.radius < 0 or self.pos.x + self.radius > WIDTH:
            self.velocity.x *= -1
        if self.pos.y - self.radius < 0 or self.pos.y + self.radius > HEIGHT:
            self.velocity.y *= -1

    def draw(self, surface):
        pygame.draw.circle(
            surface,
            (200, 50, 50),
            (int(self.pos.x), int(self.pos.y)),
            self.radius
        )


class Agent:
    def __init__(self):
        self.pos = Vector2(100, 100)
        self.radius = 15
        self.heading = 0.0
        self.speed = 3.0

    def move(self, action_steer):
        self.heading += action_steer * 5.0
        forward = Vector2(1, 0).rotate(self.heading)
        self.pos += forward * self.speed

    def draw(self, surface):
        pygame.draw.circle(
            surface,
            (50, 150, 250),
            (int(self.pos.x), int(self.pos.y)),
            self.radius
        )
        forward = Vector2(1, 0).rotate(self.heading)
        end_pos = self.pos + forward * (self.radius + 20)
        pygame.draw.line(surface, (0, 255, 0), self.pos, end_pos, 2)


class SimulationEnv:
    def __init__(self, seed=None):
        """
        Python 原生环境。

        seed:
            - None: 使用非固定随机序列，适合训练。
            - int: 使用固定随机序列，适合最终 fixed-seed evaluation。
        """
        self.seed_value = seed
        self.rng = random.Random(seed)

        self.prev_distance = 0
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("GA-RL Autonomous Navigation Env")
        self.clock = pygame.time.Clock()

        self.agent = Agent()
        self.obstacles = []
        self.target_pos = Vector2()

        self.episode_start_pos = Vector2()
        self.episode_end_pos = Vector2()
        self.optimal_distance = 0.0
        self.actual_path_length = 0.0

    def seed(self, seed=None):
        """
        重设环境随机种子。

        返回 list 是为了兼容部分 RL 环境接口习惯。
        """
        self.seed_value = seed
        self.rng = random.Random(seed)
        return [seed]

    def set_seed(self, seed=None):
        return self.seed(seed)

    def reset(self, seed=None):
        """
        自动回合重置。

        若传入 seed，则先重置随机数生成器；若不传，则沿用当前 rng 的后续随机序列。
        因此：同一个 seed + 同样 episode 数量，会得到可复现的环境序列。
        """
        if seed is not None:
            self.seed(seed)

        self.agent = Agent()
        self.agent.pos = Vector2(
            self.rng.randint(50, 150),
            self.rng.randint(50, 550)
        )
        self.agent.heading = self.rng.uniform(0, 360)

        self.target_pos = Vector2(
            self.rng.randint(650, 750),
            self.rng.randint(50, 550)
        )

        self.obstacles = []

        self.episode_start_pos = self.agent.pos.copy()
        self.episode_end_pos = self.target_pos.copy()
        self.optimal_distance = self.episode_start_pos.distance_to(self.episode_end_pos)
        self.actual_path_length = 0.0

        while len(self.obstacles) < 15:
            obs = DynamicObstacle(rng=self.rng)
            safe_from_agent = (
                obs.pos.distance_to(self.agent.pos)
                > self.agent.radius + obs.radius + 80
            )
            safe_from_target = (
                obs.pos.distance_to(self.target_pos)
                > obs.radius + 50
            )
            if safe_from_agent and safe_from_target:
                self.obstacles.append(obs)

        self.prev_distance = self.agent.pos.distance_to(self.target_pos)
        return self._get_state()

    def _cast_rays(self, draw=False):
        ray_distances = []
        start_angle = -FOV / 2
        angle_step = FOV / (NUM_RAYS - 1)

        for i in range(NUM_RAYS):
            ray_angle = self.agent.heading + start_angle + (i * angle_step)
            ray_dir = Vector2(1, 0).rotate(ray_angle)

            distance = MAX_RAY_DIST

            for obs in self.obstacles:
                hit_dist = ray_to_obstacle(
                    self.agent.pos,
                    ray_dir,
                    obs.pos,
                    obs.radius,
                    distance
                )
                if hit_dist is not None and hit_dist < distance:
                    distance = hit_dist

            boundary_dist = ray_boundary_distance(
                self.agent.pos,
                ray_dir,
                WIDTH,
                HEIGHT,
                MAX_RAY_DIST
            )

            if boundary_dist is not None and boundary_dist < distance:
                distance = boundary_dist

            normalized_dist = distance / MAX_RAY_DIST
            ray_distances.append(normalized_dist)

            if draw:
                end_point = self.agent.pos + ray_dir * distance
                pygame.draw.line(
                    self.screen,
                    (0, 255, 0),
                    self.agent.pos,
                    end_point,
                    1
                )

        return ray_distances

    def _get_state(self):
        rays = self._cast_rays(draw=False)

        dir_to_target = self.target_pos - self.agent.pos
        target_angle = math.degrees(math.atan2(dir_to_target.y, dir_to_target.x))
        angle_diff = (target_angle - self.agent.heading + 180) % 360 - 180
        normalized_yaw = angle_diff / 180.0

        normalized_target_dist = min(
            self.agent.pos.distance_to(self.target_pos) / math.hypot(WIDTH, HEIGHT),
            1.0
        )

        state = rays + [normalized_yaw, normalized_target_dist]
        return state

    def step(self, action_steer):
        """执行单步动作，更新环境并返回 next_state, reward, done, info。"""
        previous_pos = self.agent.pos.copy()
        self.agent.move(action_steer)
        step_distance = previous_pos.distance_to(self.agent.pos)
        self.actual_path_length += step_distance

        for obs in self.obstacles:
            obs.update()

        done = False
        termination_reason = None

        current_distance = self.agent.pos.distance_to(self.target_pos)
        distance_delta = (
            self.prev_distance - current_distance
        ) / math.hypot(WIDTH, HEIGHT)
        reward = distance_delta * 100.0
        reward -= 0.01
        self.prev_distance = current_distance

        # 1. 障碍物碰撞死亡机制
        for obs in self.obstacles:
            if self.agent.pos.distance_to(obs.pos) < (self.agent.radius + obs.radius - 10):
                done = True
                termination_reason = "obstacle_collision"
                reward = -50.0
                break

        # 2. 边界碰撞死亡机制
        if not done:
            if (
                self.agent.pos.x - self.agent.radius < 0 or
                self.agent.pos.x + self.agent.radius > WIDTH or
                self.agent.pos.y - self.agent.radius < 0 or
                self.agent.pos.y + self.agent.radius > HEIGHT
            ):
                done = True
                termination_reason = "boundary_collision"
                reward = -50.0

        # 3. 目标到达检测
        if not done:
            if current_distance < self.agent.radius + 15:
                done = True
                termination_reason = "success"
                reward = 100.0

        next_state = self._get_state()

        path_efficiency_ratio = None
        if termination_reason == "success" and self.optimal_distance > 0:
            path_efficiency_ratio = self.actual_path_length / self.optimal_distance

        info = {
            "seed": self.seed_value,
            "termination_reason": termination_reason,
            "collision": termination_reason in ["obstacle_collision", "boundary_collision"],
            "success": termination_reason == "success",
            "actual_path_length": self.actual_path_length,
            "optimal_distance": self.optimal_distance,
            "path_efficiency_ratio": path_efficiency_ratio
        }

        return next_state, reward, done, info

    def render(self):
        """支持 60FPS 渲染。"""
        self.screen.fill((30, 30, 30))

        pygame.draw.circle(
            self.screen,
            (50, 200, 50),
            (int(self.target_pos.x), int(self.target_pos.y)),
            15
        )

        for obs in self.obstacles:
            obs.draw(self.screen)

        self.agent.draw(self.screen)
        self._cast_rays(draw=True)

        pygame.display.flip()
        self.clock.tick(FPS)


if __name__ == "__main__":
    env = SimulationEnv(seed=0)
    state = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.rng.choice([-1, 0, 1])
        next_state, reward, done, info = env.step(action)

        if done:
            print("------------------------------------------------------------------------------------------------")
            env.reset()

        env.render()
        pygame.display.flip()

    pygame.quit()
