


from pathlib import Path
import csv
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common

from Models.TFEnv import TFAgentSimulationEnv
from Models.Utils import Utils_stats


class DQNEvaluator:
    """
    DQN 最终评估模块。

    职责：
    1. 重建与训练时相同的 TF-Agents 环境；
    2. 重建与训练时相同的 QNetwork 和 DqnAgent；
    3. 从 checkpoint 恢复训练好的权重；
    4. 使用 greedy policy 运行最终评估；
    5. 统计最终策略的：
       - Average Return
       - Success Rate
       - Collision Rate
       - Obstacle Collision Rate
       - Boundary Collision Rate
       - Steps to Success
       - Path Efficiency

    不负责：
    - Average Return over Training Steps
    - Early Collision Rate over first 500 training episodes
    - Collision Rate Decay Curve
    """

    def __init__(
        self,
        checkpoint_dir,
        fc_layer_params=(64, 64),
        learning_rate=5e-4
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fc_layer_params = fc_layer_params
        self.learning_rate = learning_rate

        self.utils = Utils_stats()

        self.eval_py_env = None
        self.eval_tf_env = None
        self.agent = None
        self.train_step_counter = None
        self.checkpointer = None

        self._build_and_restore()

    def _build_and_restore(self):
        """
        重建环境、Q 网络、DQN agent，并从 checkpoint 恢复参数。
        """

        self.eval_py_env = TFAgentSimulationEnv()
        self.eval_tf_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        q_net = q_network.QNetwork(
            self.eval_tf_env.observation_spec(),
            self.eval_tf_env.action_spec(),
            fc_layer_params=self.fc_layer_params
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.eval_tf_env.time_step_spec(),
            self.eval_tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_huber_loss,
            train_step_counter=self.train_step_counter
        )

        self.agent.initialize()

        self.checkpointer = common.Checkpointer(
            ckpt_dir=str(self.checkpoint_dir),
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            global_step=self.train_step_counter
        )

        print(f"Checkpoint restored from: {self.checkpoint_dir}")
        print(f"Restored training step: {self.train_step_counter.numpy()}")

    def reset_eval_records(self):
        """
        清空评估环境中的 episode records。
        每次独立评估前都应该调用，避免混入旧数据。
        """

        self.eval_py_env.episode_records.clear()
        self.eval_py_env.episode_count = 0
        self.eval_py_env.collision_count = 0
        self.eval_py_env.success_count = 0

    def evaluate_policy(
        self,
        num_episodes=100,
        max_steps_per_episode=1000,
        clear_records=True
    ) -> Dict[str, Any]:
        """
        使用训练好的 greedy policy 运行最终评估。

        返回：
            一个 summary 字典，包含最终评估指标。
        """

        if clear_records:
            self.reset_eval_records()

        avg_return = self.utils.compute_avg_return(
            environment=self.eval_tf_env,
            policy=self.agent.policy,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode
        )

        summary = self._build_summary(
            avg_return=avg_return
        )

        return summary

    def _build_summary(self, avg_return: float) -> Dict[str, Any]:
        """
        从 eval_py_env.episode_records 中汇总最终评估指标。
        """

        records = self.eval_py_env.episode_records
        num_episodes = len(records)

        if num_episodes == 0:
            return {
                "num_episodes": 0,
                "avg_return": avg_return,
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "obstacle_collision_rate": 0.0,
                "boundary_collision_rate": 0.0,
                "steps_to_success": self._empty_steps_to_success(),
                "path_efficiency": self._empty_path_efficiency()
            }

        success_rate = self.utils.compute_success_rate(self.eval_py_env)
        path_efficiency = self.utils.compute_path_efficiency(self.eval_py_env)

        collision_stats = self._compute_collision_stats(records)
        steps_to_success = self._compute_steps_to_success(records)

        return {
            "num_episodes": num_episodes,
            "avg_return": float(avg_return),
            "success_rate": success_rate,
            "collision_rate": collision_stats["collision_rate"],
            "obstacle_collision_rate": collision_stats["obstacle_collision_rate"],
            "boundary_collision_rate": collision_stats["boundary_collision_rate"],
            "collision_count": collision_stats["collision_count"],
            "obstacle_collision_count": collision_stats["obstacle_collision_count"],
            "boundary_collision_count": collision_stats["boundary_collision_count"],
            "success_count": collision_stats["success_count"],
            "steps_to_success": steps_to_success,
            "path_efficiency": path_efficiency
        }

    def _compute_collision_stats(self, records):
        """
        统计最终评估阶段的碰撞率。
        """

        num_episodes = len(records)

        collision_count = sum(
            int(record.get("collision", 0))
            for record in records
        )

        success_count = sum(
            int(record.get("success", 0))
            for record in records
        )

        obstacle_collision_count = sum(
            1 for record in records
            if record.get("termination_reason") == "obstacle_collision"
        )

        boundary_collision_count = sum(
            1 for record in records
            if record.get("termination_reason") == "boundary_collision"
        )

        return {
            "collision_count": collision_count,
            "success_count": success_count,
            "obstacle_collision_count": obstacle_collision_count,
            "boundary_collision_count": boundary_collision_count,
            "collision_rate": collision_count / num_episodes,
            "obstacle_collision_rate": obstacle_collision_count / num_episodes,
            "boundary_collision_rate": boundary_collision_count / num_episodes
        }

    def _compute_steps_to_success(self, records):
        """
        只统计 success episode 的 steps。
        失败 episode 不参与 Steps to Success 均值计算。
        """

        success_steps = []

        for record in records:
            if int(record.get("success", 0)) != 1:
                continue

            steps = record.get("steps", None)

            if steps is None or steps == "":
                continue

            success_steps.append(float(steps))

        if len(success_steps) == 0:
            return self._empty_steps_to_success()

        success_steps = np.array(success_steps, dtype=np.float32)

        return {
            "success_count": int(len(success_steps)),
            "mean_steps": float(np.mean(success_steps)),
            "median_steps": float(np.median(success_steps)),
            "min_steps": float(np.min(success_steps)),
            "max_steps": float(np.max(success_steps))
        }

    def _empty_steps_to_success(self):
        return {
            "success_count": 0,
            "mean_steps": None,
            "median_steps": None,
            "min_steps": None,
            "max_steps": None
        }

    def _empty_path_efficiency(self):
        return {
            "success_count": 0,
            "mean_path_efficiency": None,
            "median_path_efficiency": None,
            "min_path_efficiency": None,
            "max_path_efficiency": None
        }

    def save_eval_records(self, save_path):
        """
        保存最终评估阶段 episode records。
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "episode",
            "collision",
            "success",
            "termination_reason",
            "steps",
            "episode_return",
            "actual_path_length",
            "optimal_distance",
            "path_efficiency_ratio"
        ]

        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in self.eval_py_env.episode_records:
                row = {field: record.get(field, None) for field in fieldnames}
                writer.writerow(row)

        print(f"Final evaluation records saved to: {save_path}")

    def print_summary(self, summary: Dict[str, Any], prefix="Final DQN Policy Evaluation"):
        """
        打印最终评估结果。
        """

        print(f"\n[{prefix}]")
        print(f"Num Episodes: {summary['num_episodes']}")
        print(f"Avg Return: {summary['avg_return']:.3f}")

        print(f"Success Rate: {summary['success_rate']:.3f}")
        print(f"Collision Rate: {summary['collision_rate']:.3f}")
        print(f"Obstacle Collision Rate: {summary['obstacle_collision_rate']:.3f}")
        print(f"Boundary Collision Rate: {summary['boundary_collision_rate']:.3f}")

        print(f"Success Count: {summary['success_count']}")
        print(f"Collision Count: {summary['collision_count']}")
        print(f"Obstacle Collision Count: {summary['obstacle_collision_count']}")
        print(f"Boundary Collision Count: {summary['boundary_collision_count']}")

        print(f"Steps to Success: {summary['steps_to_success']}")
        print(f"Path Efficiency: {summary['path_efficiency']}")