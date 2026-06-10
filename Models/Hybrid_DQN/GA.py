import os
import sys
import csv
from pathlib import Path

import numpy as np


# 如果不想弹出 pygame 窗口，保持 True。
HEADLESS = True

if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame


# ------------------------------------------------------------
# 自动查找项目根目录，确保可以 import Env.py
# ------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR

while not (PROJECT_ROOT / "Env.py").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "Env.py").exists():
    raise FileNotFoundError(
        "找不到 Env.py。请确认 GA_heuristic.py 放在 D:\\MSc_Project 或其子目录下。"
    )

sys.path.insert(0, str(PROJECT_ROOT))

from Env import SimulationEnv


# ------------------------------------------------------------
# 动作映射：必须和 DQN / TFEnv 保持一致
# ------------------------------------------------------------
ACTION_MAPPING = {
    0: -1.0,
    1: -0.5,
    2: 0.0,
    3: 0.5,
    4: 1.0
}


# ------------------------------------------------------------
# 低维启发式控制器
# ------------------------------------------------------------
"""
参数含义：

params[0] = w_target
    目标方向权重。控制智能体朝目标偏航角修正。

params[1] = w_side
    左右障碍物危险差异权重。
    left_danger > right_danger 时，说明左侧更危险，应倾向向右转。

params[2] = w_front
    正前方危险权重。
    当前方射线很短时，增强避让。

params[3] = w_front_group
    前方一组射线危险权重。
    不只看正中间一根 ray，而是看 rays[4:7] 的局部危险。

params[4] = bias
    转向偏置。

params[5] = threshold_small
    小转向阈值。

params[6] = threshold_large
    大转向阈值。

最终输出：
    0 -> -1.0 大左转
    1 -> -0.5 小左转
    2 ->  0.0 直行
    3 ->  0.5 小右转
    4 ->  1.0 大右转
"""


PARAM_BOUNDS = np.array([
    [-4.0, 4.0],    # w_target
    [-6.0, 6.0],    # w_side
    [-8.0, 8.0],    # w_front
    [-8.0, 8.0],    # w_front_group
    [-0.8, 0.8],    # bias
    [0.05, 0.50],   # threshold_small
    [0.30, 1.50],   # threshold_large
], dtype=np.float32)


def heuristic_action(state, params):
    """
    根据 13 维状态和 GA 参数输出离散动作 index: 0..4。
    """

    state = np.asarray(state, dtype=np.float32)
    params = np.asarray(params, dtype=np.float32)

    rays = np.clip(state[:11], 0.0, 1.0)
    yaw = float(np.clip(state[11], -1.0, 1.0))
    target_dist = float(np.clip(state[12], 0.0, 1.0))

    w_target = params[0]
    w_side = params[1]
    w_front = params[2]
    w_front_group = params[3]
    bias = params[4]
    threshold_small = abs(float(params[5]))
    threshold_large = abs(float(params[6]))

    if threshold_large <= threshold_small:
        threshold_large = threshold_small + 0.1

    # rays: 0..4 左侧视野，5 正前方，6..10 右侧视野
    left_rays = rays[:5]
    front_ray = rays[5]
    right_rays = rays[6:]

    left_danger = float(np.mean(1.0 - left_rays))
    right_danger = float(np.mean(1.0 - right_rays))
    front_danger = float(1.0 - front_ray)
    front_group_danger = float(np.mean(1.0 - rays[4:7]))

    # 左侧更危险时，left_danger - right_danger > 0
    side_balance = left_danger - right_danger

    # 前方危险时，选择一个侧向避让方向。
    # 如果左右危险差明显，就朝相对安全侧转；
    # 如果左右差不明显，就参考目标方向 yaw。
    if abs(side_balance) > 1e-6:
        escape_sign = np.sign(side_balance)
    else:
        escape_sign = np.sign(yaw) if abs(yaw) > 1e-6 else 0.0

    # 目标项：朝目标修正
    target_term = w_target * yaw

    # 障碍物侧向项：左右危险不平衡时转向
    side_term = w_side * side_balance

    # 前方危险项：前方越危险，越需要转向
    front_term = w_front * front_danger * escape_sign

    # 前方区域危险项：考虑正前方附近多根射线
    front_group_term = w_front_group * front_group_danger * escape_sign

    # 目标越远，越允许明显转向避障；目标越近，目标项影响更大
    distance_factor = 0.5 + target_dist

    steer_score = (
        bias
        + target_term
        + distance_factor * (side_term + front_term + front_group_term)
    )

    if steer_score < -threshold_large:
        return 0
    elif steer_score < -threshold_small:
        return 1
    elif steer_score > threshold_large:
        return 4
    elif steer_score > threshold_small:
        return 3
    else:
        return 2


# ------------------------------------------------------------
# 单个 episode 运行
# ------------------------------------------------------------
def run_episode(params, max_steps=500, render=False):
    env = SimulationEnv()
    state = env.reset()

    initial_dist = env.agent.pos.distance_to(env.target_pos)
    min_dist = initial_dist

    done = False
    info = {}
    step_count = 0
    episode_return = 0.0

    while not done and step_count < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action_index = heuristic_action(state, params)
        real_action = ACTION_MAPPING[action_index]

        state, reward, done, info = env.step(real_action)

        episode_return += float(reward)
        step_count += 1

        current_dist = env.agent.pos.distance_to(env.target_pos)
        min_dist = min(min_dist, current_dist)

        if render:
            env.render()

    final_dist = env.agent.pos.distance_to(env.target_pos)
    distance_progress = initial_dist - min_dist

    if not done:
        termination_reason = "timeout"
        collision = False
        success = False
        path_efficiency_ratio = None
    else:
        termination_reason = info.get("termination_reason", "unknown")
        collision = bool(info.get("collision", False))
        success = bool(info.get("success", False))
        path_efficiency_ratio = info.get("path_efficiency_ratio", None)

    return {
        "episode_return": episode_return,
        "distance_progress": float(distance_progress),
        "final_dist": float(final_dist),
        "steps": step_count,
        "termination_reason": termination_reason,
        "collision": collision,
        "success": success,
        "path_efficiency_ratio": path_efficiency_ratio
    }


# ------------------------------------------------------------
# Fitness 函数
# ------------------------------------------------------------
def objective(params, eval_episodes=5, max_steps=500):
    """
    GA 适应度函数。

    目标不是让 GA 成为最终最强策略，而是找到：
    1. 比随机策略更安全；
    2. early collision 更低；
    3. 能产生一定成功轨迹；
    4. 可用于 DQN replay buffer warm-start 的启发式控制器。
    """

    total_score = 0.0

    for _ in range(eval_episodes):
        result = run_episode(
            params=params,
            max_steps=max_steps,
            render=False
        )

        score = 0.0

        # 1. 使用环境 return 作为基础
        score += result["episode_return"]

        # 2. 鼓励接近目标
        score += result["distance_progress"] * 0.5

        # 3. 成功奖励
        if result["success"]:
            score += 300.0
            score += (max_steps - result["steps"]) * 0.5

        # 4. 碰撞惩罚
        elif result["termination_reason"] == "obstacle_collision":
            score -= 180.0

        elif result["termination_reason"] == "boundary_collision":
            score -= 180.0

        # 5. timeout 没撞但没成功，按最终距离惩罚
        else:
            score -= result["final_dist"] * 0.05

        total_score += score

    return total_score / eval_episodes


# ------------------------------------------------------------
# GA 基本操作
# ------------------------------------------------------------
def init_population(n_pop, bounds):
    low = bounds[:, 0]
    high = bounds[:, 1]

    return np.random.uniform(
        low=low,
        high=high,
        size=(n_pop, len(bounds))
    ).astype(np.float32)


def tournament_selection(pop, scores, k=3):
    selected_ix = np.random.randint(len(pop))

    for ix in np.random.randint(0, len(pop), size=k - 1):
        if scores[ix] > scores[selected_ix]:
            selected_ix = ix

    return pop[selected_ix].copy()


def arithmetic_crossover(p1, p2, cross_rate=0.9):
    c1 = p1.copy()
    c2 = p2.copy()

    if np.random.rand() < cross_rate:
        alpha = np.random.rand()

        c1 = alpha * p1 + (1.0 - alpha) * p2
        c2 = alpha * p2 + (1.0 - alpha) * p1

    return c1.astype(np.float32), c2.astype(np.float32)


def gaussian_mutation(individual, bounds, mutation_rate=0.3, sigma=0.15):
    child = individual.copy()

    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] += np.random.normal(0.0, sigma)

    child = np.clip(child, bounds[:, 0], bounds[:, 1])

    return child.astype(np.float32)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def evaluate_controller(params, num_episodes=100, max_steps=500, csv_path=None):
    records = []
    total_return = 0.0

    for ep in range(1, num_episodes + 1):
        result = run_episode(
            params=params,
            max_steps=max_steps,
            render=False
        )

        record = {
            "episode": ep,
            "termination_reason": result["termination_reason"],
            "collision": int(result["collision"]),
            "success": int(result["success"]),
            "steps": result["steps"],
            "episode_return": result["episode_return"],
            "path_efficiency_ratio": result["path_efficiency_ratio"]
        }

        records.append(record)
        total_return += result["episode_return"]

    success_count = sum(r["success"] for r in records)
    collision_count = sum(r["collision"] for r in records)

    obstacle_collision_count = sum(
        1 for r in records
        if r["termination_reason"] == "obstacle_collision"
    )

    boundary_collision_count = sum(
        1 for r in records
        if r["termination_reason"] == "boundary_collision"
    )

    timeout_count = sum(
        1 for r in records
        if r["termination_reason"] == "timeout"
    )

    path_efficiencies = [
        float(r["path_efficiency_ratio"])
        for r in records
        if r["success"] == 1 and r["path_efficiency_ratio"] is not None
    ]

    summary = {
        "num_episodes": num_episodes,
        "avg_return": total_return / num_episodes,
        "success_rate": success_count / num_episodes,
        "collision_rate": collision_count / num_episodes,
        "obstacle_collision_rate": obstacle_collision_count / num_episodes,
        "boundary_collision_rate": boundary_collision_count / num_episodes,
        "timeout_rate": timeout_count / num_episodes,
        "success_count": success_count,
        "collision_count": collision_count,
        "mean_path_efficiency": None if len(path_efficiencies) == 0 else float(np.mean(path_efficiencies)),
        "median_path_efficiency": None if len(path_efficiencies) == 0 else float(np.median(path_efficiencies))
    }

    if csv_path is not None:
        save_records_to_csv(records, csv_path)

    return summary


def save_records_to_csv(records, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "episode",
        "termination_reason",
        "collision",
        "success",
        "steps",
        "episode_return",
        "path_efficiency_ratio"
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ------------------------------------------------------------
# GA 主循环
# ------------------------------------------------------------
def genetic_algorithm_heuristic(
    bounds,
    n_pop=80,
    n_it=60,
    cross_rate=0.9,
    mutation_rate=0.35,
    sigma=0.20,
    elite_size=2,
    eval_episodes=5,
    validation_interval=5,
    save_dir=None
):
    pop = init_population(n_pop, bounds)

    scores = np.array([
        objective(ind, eval_episodes=eval_episodes)
        for ind in pop
    ], dtype=np.float32)

    best_ix = int(np.argmax(scores))
    best = pop[best_ix].copy()
    best_score = float(scores[best_ix])

    history = []

    print(f"初始最优得分: {best_score:.2f}")
    print(f"初始最优参数: {best}")

    for gen in range(1, n_it + 1):
        ranked_indices = np.argsort(scores)[::-1]

        gen_best_score = float(scores[ranked_indices[0]])
        gen_mean_score = float(np.mean(scores))

        if gen_best_score > best_score:
            best_ix = ranked_indices[0]
            best = pop[best_ix].copy()
            best_score = gen_best_score
            print(f"> 第 {gen} 代发现新最优 | score={best_score:.2f} | params={best}")

        validation = None

        if gen == 1 or gen % validation_interval == 0:
            validation = evaluate_controller(
                best,
                num_episodes=30,
                max_steps=500
            )

            print(
                f"[Validation Gen {gen}] "
                f"Success={validation['success_rate']:.2f} | "
                f"Collision={validation['collision_rate']:.2f} | "
                f"Obstacle={validation['obstacle_collision_rate']:.2f} | "
                f"Boundary={validation['boundary_collision_rate']:.2f} | "
                f"AvgReturn={validation['avg_return']:.2f}"
            )

        history.append({
            "generation": gen,
            "gen_best_score": gen_best_score,
            "gen_mean_score": gen_mean_score,
            "global_best_score": best_score,
            "sigma": sigma,
            "best_params": best.tolist(),
            "validation_success_rate": None if validation is None else validation["success_rate"],
            "validation_collision_rate": None if validation is None else validation["collision_rate"],
            "validation_obstacle_collision_rate": None if validation is None else validation["obstacle_collision_rate"],
            "validation_boundary_collision_rate": None if validation is None else validation["boundary_collision_rate"],
            "validation_avg_return": None if validation is None else validation["avg_return"]
        })

        elites = [
            pop[ix].copy()
            for ix in ranked_indices[:elite_size]
        ]

        selected = [
            tournament_selection(pop, scores, k=3)
            for _ in range(n_pop)
        ]

        children = []
        children.extend(elites)

        while len(children) < n_pop:
            p1 = selected[np.random.randint(0, n_pop)]
            p2 = selected[np.random.randint(0, n_pop)]

            c1, c2 = arithmetic_crossover(
                p1,
                p2,
                cross_rate=cross_rate
            )

            c1 = gaussian_mutation(
                c1,
                bounds=bounds,
                mutation_rate=mutation_rate,
                sigma=sigma
            )

            c2 = gaussian_mutation(
                c2,
                bounds=bounds,
                mutation_rate=mutation_rate,
                sigma=sigma
            )

            children.append(c1)

            if len(children) < n_pop:
                children.append(c2)

        pop = np.asarray(children, dtype=np.float32)

        scores = np.array([
            objective(ind, eval_episodes=eval_episodes)
            for ind in pop
        ], dtype=np.float32)

        sigma *= 0.98

        print(
            f"第 {gen}/{n_it} 代 | "
            f"GenBest={gen_best_score:.2f} | "
            f"GenMean={gen_mean_score:.2f} | "
            f"GlobalBest={best_score:.2f} | "
            f"Sigma={sigma:.4f}"
        )

    if save_dir is not None:
        save_history(history, Path(save_dir) / "GA_Heuristic_Training_History.csv")

    return best, best_score, history


def save_history(history, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "generation",
        "gen_best_score",
        "gen_mean_score",
        "global_best_score",
        "sigma",
        "best_params",
        "validation_success_rate",
        "validation_collision_rate",
        "validation_obstacle_collision_rate",
        "validation_boundary_collision_rate",
        "validation_avg_return"
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"GA history saved to: {csv_path}")


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    run_dir = CURRENT_DIR / "runs" / "ga_heuristic_controller"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = run_dir / "best_ga_heuristic_params.npy"
    final_eval_csv_path = run_dir / "GA_Heuristic_Final_Evaluation.csv"

    print("开始 GA 进化低维启发式控制器参数。")
    print(f"参数维度: {len(PARAM_BOUNDS)}")
    print(f"参数边界:\n{PARAM_BOUNDS}")

    best_params, best_score, history = genetic_algorithm_heuristic(
        bounds=PARAM_BOUNDS,
        n_pop=80,
        n_it=60,
        cross_rate=0.9,
        mutation_rate=0.35,
        sigma=0.20,
        elite_size=2,
        eval_episodes=5,
        validation_interval=5,
        save_dir=run_dir
    )

    np.save(best_params_path, best_params.astype(np.float32))

    print("\n进化完成。")
    print(f"Best score: {best_score:.2f}")
    print(f"Best params: {best_params}")
    print(f"Best params saved to: {best_params_path}")

    final_summary = evaluate_controller(
        best_params,
        num_episodes=100,
        max_steps=500,
        csv_path=final_eval_csv_path
    )

    print("\n[Final GA Heuristic Evaluation]")
    print(f"Num Episodes: {final_summary['num_episodes']}")
    print(f"Avg Return: {final_summary['avg_return']:.3f}")
    print(f"Success Rate: {final_summary['success_rate']:.3f}")
    print(f"Collision Rate: {final_summary['collision_rate']:.3f}")
    print(f"Obstacle Collision Rate: {final_summary['obstacle_collision_rate']:.3f}")
    print(f"Boundary Collision Rate: {final_summary['boundary_collision_rate']:.3f}")
    print(f"Timeout Rate: {final_summary['timeout_rate']:.3f}")
    print(f"Success Count: {final_summary['success_count']}")
    print(f"Collision Count: {final_summary['collision_count']}")
    print(f"Mean Path Efficiency: {final_summary['mean_path_efficiency']}")
    print(f"Median Path Efficiency: {final_summary['median_path_efficiency']}")

    print(f"Final evaluation records saved to: {final_eval_csv_path}")