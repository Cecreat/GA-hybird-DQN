from pathlib import Path
import csv
import random
from typing import Dict, Any, Iterable, Optional

import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common

try:
    from Models.TFEnv import TFAgentSimulationEnv
    from Models.Utils import Utils_stats
except ModuleNotFoundError:
    from Models.TFEnv import TFAgentSimulationEnv
    from Models.Utils import Utils_stats


class DQNEvaluator:
    """
    DQN 最终评估模块。

    职责：
    1. 重建与训练时相同的 TF-Agents 环境；
    2. 重建与训练时相同的 QNetwork 和 DqnAgent；
    3. 从指定 checkpoint 目录恢复训练好的权重；
    4. 使用 greedy policy 运行最终评估；
    5. 支持普通评估与 fixed-seed repeated evaluation；
    6. 统计最终策略的 Average Return、Success Rate、Collision Rate、
       Obstacle Collision Rate、Boundary Collision Rate、Steps to Success、
       Path Efficiency。

    用法：
        最终评估时建议传入训练模块返回的 best_checkpoint_dir，
        而不是 latest checkpoint_dir。
    """

    def __init__(
        self,
        checkpoint_dir,
        fc_layer_params=(64, 64),
        learning_rate=1e-4
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

    def _make_eval_env(self, seed: Optional[int] = None):
        """
        创建评估环境。

        每组 fixed seed 都重新创建一个独立环境，避免不同 seed 的 episode_records
        或随机序列互相污染。
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        eval_py_env = TFAgentSimulationEnv(seed=seed)
        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        return eval_py_env, eval_tf_env

    def _build_and_restore(self):
        """
        重建环境、Q 网络、DQN agent，并从 checkpoint 恢复参数。
        """
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"checkpoint 目录不存在: {self.checkpoint_dir}"
            )

        latest_checkpoint = tf.train.latest_checkpoint(str(self.checkpoint_dir))
        if latest_checkpoint is None:
            raise FileNotFoundError(
                f"没有在该目录找到可恢复的 checkpoint: {self.checkpoint_dir}"
            )

        self.eval_py_env, self.eval_tf_env = self._make_eval_env(seed=None)

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

        self.checkpointer.initialize_or_restore()

        print(f"Checkpoint restored from: {self.checkpoint_dir}")
        print(f"Latest checkpoint file: {latest_checkpoint}")
        print(f"Restored training step: {self.train_step_counter.numpy()}")

    def reset_eval_records(self):
        """
        清空评估环境中的 episode records。
        每次独立评估前都应该调用，避免混入旧数据。
        """
        if hasattr(self.eval_py_env, "reset_episode_records"):
            self.eval_py_env.reset_episode_records()
            return

        self.eval_py_env.episode_records.clear()
        self.eval_py_env.episode_count = 0
        self.eval_py_env.collision_count = 0
        self.eval_py_env.success_count = 0

    def evaluate_policy(
        self,
        num_episodes=100,
        max_steps_per_episode=1000,
        clear_records=True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        使用训练好的 greedy policy 运行一次评估。

        seed:
            - None: 普通随机评估。
            - int: 使用该 seed 重建评估环境，得到可复现的评估序列。
        """
        if seed is not None:
            self.eval_py_env, self.eval_tf_env = self._make_eval_env(seed=seed)
        elif clear_records:
            self.reset_eval_records()

        if clear_records and seed is not None:
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
        summary["seed"] = seed

        return summary

    def evaluate_policy_fixed_seeds(
        self,
        seed_list: Optional[Iterable[int]] = None,
        episodes_per_seed=500,
        max_steps_per_episode=1000,
        save_dir=None,
        save_episode_records=True
    ) -> Dict[str, Any]:
        """
        固定随机种子重复评估。

        返回结构：
            {
                "seed_list": [...],
                "episodes_per_seed": 500,
                "total_episodes": len(seed_list) * 500,
                "per_seed": [每个 seed 的扁平 summary],
                "mean_std": {metric: {"mean": x, "std": y}}
            }
        """
        if seed_list is None:
            seed_list = [0, 1, 2, 3, 4]

        seed_list = [int(seed) for seed in seed_list]

        save_dir = Path(save_dir) if save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        per_seed_rows = []

        for seed in seed_list:
            summary = self.evaluate_policy(
                num_episodes=episodes_per_seed,
                max_steps_per_episode=max_steps_per_episode,
                clear_records=True,
                seed=seed
            )

            flat_summary = self._flatten_summary(summary)
            per_seed_rows.append(flat_summary)

            print(
                f"[Fixed Seed Evaluation] seed={seed} | "
                f"episodes={episodes_per_seed} | "
                f"success_rate={flat_summary['success_rate']:.3f} | "
                f"collision_rate={flat_summary['collision_rate']:.3f} | "
                f"avg_return={flat_summary['avg_return']:.3f}"
            )

            if save_dir is not None and save_episode_records:
                self.save_eval_records(
                    save_dir / f"Evaluation_Records_seed_{seed}.csv"
                )

        mean_std = self._compute_mean_std(per_seed_rows)

        result = {
            "seed_list": seed_list,
            "episodes_per_seed": int(episodes_per_seed),
            "total_episodes": int(len(seed_list) * episodes_per_seed),
            "per_seed": per_seed_rows,
            "mean_std": mean_std
        }

        if save_dir is not None:
            self._save_rows_csv(
                per_seed_rows,
                save_dir / "FixedSeed_Evaluation_PerSeed_Summary.csv"
            )
            self._save_mean_std_csv(
                mean_std,
                save_dir / "FixedSeed_Evaluation_MeanStd.csv"
            )

        return result

    def _build_summary(self, avg_return: float) -> Dict[str, Any]:
        """
        从 eval_py_env.episode_records 中汇总最终评估指标。
        """
        records = self.eval_py_env.episode_records
        num_episodes = len(records)

        if num_episodes == 0:
            return {
                "num_episodes": 0,
                "avg_return": float(avg_return),
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "obstacle_collision_rate": 0.0,
                "boundary_collision_rate": 0.0,
                "collision_count": 0,
                "obstacle_collision_count": 0,
                "boundary_collision_count": 0,
                "success_count": 0,
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

    @staticmethod
    def _flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        steps = summary.get("steps_to_success", {}) or {}
        path = summary.get("path_efficiency", {}) or {}

        return {
            "seed": summary.get("seed", None),
            "num_episodes": summary.get("num_episodes", None),
            "avg_return": summary.get("avg_return", None),
            "success_rate": summary.get("success_rate", None),
            "collision_rate": summary.get("collision_rate", None),
            "obstacle_collision_rate": summary.get("obstacle_collision_rate", None),
            "boundary_collision_rate": summary.get("boundary_collision_rate", None),
            "success_count": summary.get("success_count", None),
            "collision_count": summary.get("collision_count", None),
            "obstacle_collision_count": summary.get("obstacle_collision_count", None),
            "boundary_collision_count": summary.get("boundary_collision_count", None),
            "steps_success_count": steps.get("success_count", None),
            "mean_steps_to_success": steps.get("mean_steps", None),
            "median_steps_to_success": steps.get("median_steps", None),
            "min_steps_to_success": steps.get("min_steps", None),
            "max_steps_to_success": steps.get("max_steps", None),
            "path_success_count": path.get("success_count", None),
            "mean_path_efficiency": path.get("mean_path_efficiency", None),
            "median_path_efficiency": path.get("median_path_efficiency", None),
            "min_path_efficiency": path.get("min_path_efficiency", None),
            "max_path_efficiency": path.get("max_path_efficiency", None)
        }

    @staticmethod
    def _compute_mean_std(rows):
        if len(rows) == 0:
            return {}

        mean_std = {}
        keys = rows[0].keys()

        for key in keys:
            if key == "seed":
                continue

            values = []
            for row in rows:
                value = row.get(key, None)
                if value is None:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    values.append(float(value))

            if len(values) == 0:
                continue

            arr = np.array(values, dtype=np.float32)
            mean_std[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            }

        return mean_std

    @staticmethod
    def _save_rows_csv(rows, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(rows) == 0:
            return

        fieldnames = list(rows[0].keys())
        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Fixed-seed per-seed summary saved to: {save_path}")

    @staticmethod
    def _save_mean_std_csv(mean_std, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "metric": metric,
                "mean": values.get("mean", None),
                "std": values.get("std", None)
            }
            for metric, values in mean_std.items()
        ]

        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["metric", "mean", "std"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Fixed-seed mean/std summary saved to: {save_path}")

    def save_eval_records(self, save_path):
        """
        保存最终评估阶段 episode records。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "seed",
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
        打印单次评估结果。
        """
        print(f"\n[{prefix}]")
        if summary.get("seed", None) is not None:
            print(f"Seed: {summary['seed']}")
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

    def print_fixed_seed_summary(
        self,
        fixed_seed_result: Dict[str, Any],
        prefix="Fixed-Seed DQN Policy Evaluation"
    ):
        """
        打印 fixed-seed repeated evaluation 的 mean ± std。
        """
        print(f"\n[{prefix}]")
        print(f"Seeds: {fixed_seed_result['seed_list']}")
        print(f"Episodes per Seed: {fixed_seed_result['episodes_per_seed']}")
        print(f"Total Episodes: {fixed_seed_result['total_episodes']}")

        mean_std = fixed_seed_result["mean_std"]
        key_metrics = [
            "avg_return",
            "success_rate",
            "collision_rate",
            "obstacle_collision_rate",
            "boundary_collision_rate",
            "mean_steps_to_success",
            "mean_path_efficiency"
        ]

        for metric in key_metrics:
            if metric not in mean_std:
                continue
            mean = mean_std[metric]["mean"]
            std = mean_std[metric]["std"]
            print(f"{metric}: {mean:.3f} ± {std:.3f}")
