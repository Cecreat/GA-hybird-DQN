import os
import csv
import numpy as np
import tensorflow as tf
import pygame

# 导入我们写好的 TF 环境包装器，完美解决所有张量维度问题！
from Models.DQN_agent import TFAgentSimulationEnv
from tf_agents.environments import tf_py_environment


def evaluate_policy(policy_dir, model_name, num_episodes=100):
    """
    运行指定策略，收集评估数据
    """
    print(f"\n🚀 正在加载模型 [{model_name}] 从: {policy_dir} ...")
    if not os.path.exists(policy_dir):
        print(f"❌ 警告：未找到 {model_name} 的模型文件，跳过评估。")
        return []

    saved_policy = tf.saved_model.load(policy_dir)

    # 【核心修复】：使用 TFPyEnvironment 包装器自动处理 Batch 维度
    py_env = TFAgentSimulationEnv()
    env = tf_py_environment.TFPyEnvironment(py_env)

    # 获取底层的原生物理环境，用于提取距离等物理参数
    underlying_env = py_env._env

    results = []

    print(f"📊 开始进行 {num_episodes} 局高强度盲测 (后台极速运行)...")
    for episode in range(num_episodes):
        time_step = env.reset()
        survival_steps = 0

        # 记录初始距离
        initial_distance = underlying_env.agent.pos.distance_to(underlying_env.target_pos)

        # 只要游戏没结束，且存活步数没到 500 步
        while not time_step.is_last() and survival_steps < 500:
            # 清理事件队列，防止无头模式下内存溢出
            for event in pygame.event.get():
                pass

            # 策略推理与环境步进
            policy_step = saved_policy.action(time_step)
            time_step = env.step(policy_step.action)
            survival_steps += 1

        # 记录最终距离
        final_distance = underlying_env.agent.pos.distance_to(underlying_env.target_pos)

        # --- 计算 T6 要求的核心指标 ---
        # 1. 结局判定
        if final_distance < 20:
            outcome = "Goal Reached"  # 完美通关
        elif time_step.is_last():
            outcome = "Collision"  # 撞墙死亡
        else:
            outcome = "Timeout"  # 原地转圈 (Reward Hacking)

        # 2. 路径效率 (推进的距离除以花费的步数)
        distance_progress = initial_distance - final_distance
        efficiency = distance_progress / survival_steps if survival_steps > 0 else 0

        results.append({
            "Model": model_name,
            "Episode": episode + 1,
            "Survival_Steps": survival_steps,
            "Outcome": outcome,
            "Final_Distance": round(final_distance, 2),
            "Path_Efficiency": round(efficiency, 4)
        })

    # 打印本模型的统计摘要
    collisions = sum(1 for r in results if r["Outcome"] == "Collision")
    timeouts = sum(1 for r in results if r["Outcome"] == "Timeout")
    goals = sum(1 for r in results if r["Outcome"] == "Goal Reached")
    avg_steps = np.mean([r["Survival_Steps"] for r in results])
    avg_dist = np.mean([r["Final_Distance"] for r in results])

    print(f"✅ {model_name} 评估完成！")
    print(f"   -> 平均存活步数: {avg_steps:.1f} / 500")
    print(f"   -> 碰撞率 (Collision Rate): {collisions / num_episodes * 100:.1f}%")
    print(f"   -> 苟活超时率 (Timeout Rate): {timeouts / num_episodes * 100:.1f}%")
    print(f"   -> 通关率 (Goal Reached Rate): {goals / num_episodes * 100:.1f}%")
    print(f"   -> 平均最终距离: {avg_dist:.1f} 像素")

    return results


def export_to_csv(all_results, filename="T6_Evaluation_Data.csv"):
    if not all_results:
        return

    keys = all_results[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_results)
    print(f"\n💾 所有评估数据已成功导出至: {os.path.join(os.getcwd(), filename)}")


if __name__ == "__main__":
    # 强制开启无头模式（不渲染画面），极大加速评估过程
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    dqn_path = os.path.join(os.getcwd(), '../baseline_dqn_policy')
    hybrid_path = os.path.join(os.getcwd(), '../hybrid_ga_rl_policy')

    # 两个模型各跑 100 局
    dqn_results = evaluate_policy(dqn_path, "Pure DQN", num_episodes=100)
    hybrid_results = evaluate_policy(hybrid_path, "Hybrid GA-RL", num_episodes=100)

    all_data = dqn_results + hybrid_results
    export_to_csv(all_data)