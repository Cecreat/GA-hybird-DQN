import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import shutil
from pathlib import Path

import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

try:
    from Models.TFEnv import TFAgentSimulationEnv
    from Models.Utils import Utils_stats
except ModuleNotFoundError:
    from Models.TFEnv import TFAgentSimulationEnv
    from  Models.Utils import Utils_stats


BASE_DIR = Path(__file__).resolve().parent


class Train:
    def __init__(self):
        self.utils = Utils_stats()

    def train_tf_agents_dqn(self):
        # -----------------------------
        # 超参数
        # -----------------------------
        num_iterations = 200000  # 总训练迭代次数，不是 episode 数
        initial_collect_steps = 5000
        collect_steps_per_iteration = 1
        replay_buffer_max_length = 50000
        batch_size = 64
        learning_rate = 1e-4
        log_interval = 200
        eval_interval = 2000

        # checkpoint selection 使用的 validation episodes。
        # 这是训练过程中的模型选择集，不是最终测试集。
        selection_eval_episodes = 100
        selection_max_steps_per_episode = 1000

        fc_layer_params = (64, 64)

        plot_steps = []
        plot_returns = []
        monitor_records = []
        checkpoint_selection_records = []
        best_score = None
        best_selection_record = None

        # -----------------------------
        # 环境
        # -----------------------------
        py_env = TFAgentSimulationEnv()
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

        eval_py_env = TFAgentSimulationEnv()
        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        # -----------------------------
        # QNetwork 与 DQN Agent
        # -----------------------------
        q_net = q_network.QNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=fc_layer_params
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_step_counter = tf.Variable(0)

        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0,
            decay_steps=num_iterations,
            end_learning_rate=0.05
        )

        agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_huber_loss,
            train_step_counter=train_step_counter,
            epsilon_greedy=lambda: epsilon_fn(train_step_counter),
            target_update_period=500,
            gamma=0.99
        )
        agent.initialize()

        # -----------------------------
        # 输出路径
        # -----------------------------
        run_dir = BASE_DIR / "runs" / "dqn_obs13_fc64_64_act5_15obstacles_(2)"
        checkpoint_dir = run_dir / "checkpoint"                 # latest checkpoint
        candidate_checkpoint_root = run_dir / "candidate_checkpoints"
        best_checkpoint_dir = run_dir / "best_checkpoint"       # final evaluation 使用这个

        plot_path = run_dir / "Baseline_Convergence_Curve.png"
        collision_plot_path = run_dir / "Baseline_Early_Collision_Rate_Decay_Curve.png"
        episode_records_path = run_dir / "Baseline_Training_Episode_Records.csv"
        monitor_records_path = run_dir / "Baseline_Training_Monitor_Records.csv"
        checkpoint_selection_records_path = run_dir / "Baseline_Checkpoint_Selection_Records.csv"

        run_dir.mkdir(parents=True, exist_ok=True)
        self._clear_dir(candidate_checkpoint_root)
        self._clear_dir(best_checkpoint_dir)

        latest_checkpointer = common.Checkpointer(
            ckpt_dir=str(checkpoint_dir),
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            global_step=train_step_counter
        )

        # -----------------------------
        # Replay Buffer
        # -----------------------------
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_max_length
        )

        def collect_step(environment, policy, buffer):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            buffer.add_batch(traj)

        # -----------------------------
        # 初始化随机经验收集
        # -----------------------------
        print("正在进行初始化随机数据收集.....")
        random_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec()
        )
        for _ in range(initial_collect_steps):
            collect_step(tf_env, random_policy, replay_buffer)
        print("初始化收集完成，当前 replay buffer 帧数：", replay_buffer.num_frames().numpy())

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2
        ).prefetch(3)

        iterator = iter(dataset)

        agent.train = common.function(agent.train)
        agent.train_step_counter.assign(0)

        # -----------------------------
        # 训练循环
        # -----------------------------
        for _ in range(num_iterations):
            # 1. 收集训练经验
            for _ in range(collect_steps_per_iteration):
                collect_step(tf_env, agent.collect_policy, replay_buffer)

            # 2. 训练一次 Q 网络
            experience, unused_info = next(iterator)
            loss_info = agent.train(experience)
            train_loss = float(loss_info.loss.numpy())

            # 3. 训练后读取 step
            step = int(agent.train_step_counter.numpy())

            # 4. 打印 loss
            if step % log_interval == 0:
                print(f"步数(step): {step} | 损失(loss): {train_loss:.4f}")

            # 5. 训练阶段监控 + best checkpoint selection
            if step % eval_interval == 0:
                self._reset_eval_env_records(eval_py_env)

                avg_return = self.utils.compute_avg_return(
                    eval_tf_env,
                    agent.policy,
                    num_episodes=selection_eval_episodes,
                    max_steps_per_episode=selection_max_steps_per_episode
                )

                selection_summary = self._build_checkpoint_selection_summary(
                    eval_py_env=eval_py_env,
                    avg_return=avg_return
                )

                early_collision_rate = self.utils.compute_early_collision_rate(
                    py_env,
                    first_n_episodes=500
                )

                plot_steps.append(step)
                plot_returns.append(selection_summary["avg_return"])

                convergence_stats = self._compute_convergence_steps(
                    plot_steps=plot_steps,
                    plot_returns=plot_returns,
                    threshold=25.0,
                    window_size=5
                )

                monitor_records.append({
                    "step": step,
                    "train_loss": train_loss,
                    "avg_return": selection_summary["avg_return"],
                    "early_collision_rate": early_collision_rate,
                    "episode_count": py_env.episode_count,
                    "converged": int(convergence_stats["converged"]),
                    "convergence_step": convergence_stats["convergence_step"],
                    "convergence_return": convergence_stats["convergence_return"]
                })

                print(
                    f"--- Baseline 训练监控 --- step: {step} | "
                    f"Avg Return: {selection_summary['avg_return']:.2f} | "
                    f"Success Rate: {selection_summary['success_rate']:.3f} | "
                    f"Collision Rate: {selection_summary['collision_rate']:.3f} | "
                    f"Early Collision Rate: {early_collision_rate:.3f} | "
                    f"Episode Count: {py_env.episode_count}"
                )

                latest_checkpointer.save(train_step_counter)

                candidate_checkpoint_dir = candidate_checkpoint_root / f"step_{step:06d}"
                self._save_checkpoint_snapshot(
                    checkpoint_dir=candidate_checkpoint_dir,
                    agent=agent,
                    train_step_counter=train_step_counter
                )

                current_score = self._checkpoint_score(selection_summary)
                is_best = best_score is None or current_score > best_score

                if is_best:
                    best_score = current_score
                    best_selection_record = {
                        "step": step,
                        **selection_summary,
                        "checkpoint_dir": str(candidate_checkpoint_dir)
                    }

                    self._copy_checkpoint_dir(
                        src_dir=candidate_checkpoint_dir,
                        dst_dir=best_checkpoint_dir
                    )

                    print(
                        f"[Baseline Best Checkpoint Updated] step={step} | "
                        f"success_rate={selection_summary['success_rate']:.3f} | "
                        f"collision_rate={selection_summary['collision_rate']:.3f} | "
                        f"avg_return={selection_summary['avg_return']:.3f}"
                    )

                checkpoint_selection_records.append({
                    "step": step,
                    "train_loss": train_loss,
                    "avg_return": float(selection_summary["avg_return"]),
                    "success_rate": float(selection_summary["success_rate"]),
                    "collision_rate": float(selection_summary["collision_rate"]),
                    "obstacle_collision_rate": float(selection_summary["obstacle_collision_rate"]),
                    "boundary_collision_rate": float(selection_summary["boundary_collision_rate"]),
                    "success_count": int(selection_summary["success_count"]),
                    "collision_count": int(selection_summary["collision_count"]),
                    "num_episodes": int(selection_summary["num_episodes"]),
                    "checkpoint_dir": str(candidate_checkpoint_dir),
                    "is_best": int(is_best)
                })

                self.utils.plot_convergence_curve(
                    plot_steps=plot_steps,
                    plot_returns=plot_returns,
                    plot_path=plot_path,
                )

                self.utils.plot_collision_rate_curve(
                    py_env=py_env,
                    save_path=collision_plot_path,
                    first_n_episodes=500,
                    window_size=50,
                    title="Baseline Early Collision Rate Decay Curve",
                )

        # -----------------------------
        # 训练结束
        # -----------------------------
        print("\n[Baseline DQN] 训练结束，保存最终 latest checkpoint...")
        latest_checkpointer.save(train_step_counter)
        print(f"最终 latest checkpoint 已保存至: {checkpoint_dir}")

        # 如果 num_iterations 太小导致没有触发 eval_interval，则用最终模型作为 fallback best。
        if best_selection_record is None:
            final_candidate_checkpoint_dir = candidate_checkpoint_root / f"step_{int(agent.train_step_counter.numpy()):06d}"
            self._save_checkpoint_snapshot(
                checkpoint_dir=final_candidate_checkpoint_dir,
                agent=agent,
                train_step_counter=train_step_counter
            )
            self._copy_checkpoint_dir(
                src_dir=final_candidate_checkpoint_dir,
                dst_dir=best_checkpoint_dir
            )
            best_selection_record = {
                "step": int(agent.train_step_counter.numpy()),
                "checkpoint_dir": str(final_candidate_checkpoint_dir),
                "note": "No validation checkpoint was created; final checkpoint used as best fallback."
            }

        final_early_collision_rate = self.utils.compute_early_collision_rate(
            py_env,
            first_n_episodes=500
        )

        training_success_rate = self.utils.compute_success_rate(py_env)

        final_convergence_stats = self._compute_convergence_steps(
            plot_steps=plot_steps,
            plot_returns=plot_returns,
            threshold=25.0,
            window_size=5
        )

        print(f"Baseline Training Early Collision Rate: {final_early_collision_rate:.3f}")
        print(f"Baseline Training Success Rate: {training_success_rate:.3f}")
        print(f"Baseline Convergence Stats: {final_convergence_stats}")
        print(f"Baseline Best Checkpoint Dir: {best_checkpoint_dir}")
        print(f"Baseline Best Selection Record: {best_selection_record}")

        self.utils.save_episode_records(
            py_env,
            episode_records_path
        )

        self._save_monitor_records(
            monitor_records,
            monitor_records_path
        )

        self._save_checkpoint_selection_records(
            checkpoint_selection_records,
            checkpoint_selection_records_path
        )

        print(f"Baseline episode records 已保存至: {episode_records_path}")
        print(f"Baseline monitor records 已保存至: {monitor_records_path}")
        print(f"Baseline checkpoint selection records 已保存至: {checkpoint_selection_records_path}")
        print(f"Baseline convergence curve 已保存至: {plot_path}")
        print(f"Baseline collision curve 已保存至: {collision_plot_path}")

        return {
            "run_dir": run_dir,
            "checkpoint_dir": checkpoint_dir,
            "best_checkpoint_dir": best_checkpoint_dir,
            "candidate_checkpoint_root": candidate_checkpoint_root,
            "episode_records_path": episode_records_path,
            "monitor_records_path": monitor_records_path,
            "checkpoint_selection_records_path": checkpoint_selection_records_path,
            "final_step": int(agent.train_step_counter.numpy()),
            "early_collision_rate": final_early_collision_rate,
            "training_success_rate": training_success_rate,
            "convergence_stats": final_convergence_stats,
            "best_selection_record": best_selection_record
        }

    @staticmethod
    def _compute_convergence_steps(
        plot_steps,
        plot_returns,
        threshold=25.0,
        window_size=5
    ):
        if len(plot_steps) != len(plot_returns):
            raise ValueError(
                f"plot_steps 和 plot_returns 长度不一致: "
                f"{len(plot_steps)} vs {len(plot_returns)}"
            )

        if len(plot_steps) < window_size:
            return {
                "converged": False,
                "convergence_step": None,
                "convergence_return": None
            }

        for i in range(window_size - 1, len(plot_returns)):
            window_returns = plot_returns[i - window_size + 1:i + 1]

            if all(value >= threshold for value in window_returns):
                return {
                    "converged": True,
                    "convergence_step": plot_steps[i],
                    "convergence_return": sum(window_returns) / window_size
                }

        return {
            "converged": False,
            "convergence_step": None,
            "convergence_return": None
        }

    @staticmethod
    def _clear_dir(path):
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reset_eval_env_records(eval_py_env):
        eval_py_env.episode_records.clear()
        eval_py_env.episode_count = 0
        eval_py_env.collision_count = 0
        eval_py_env.success_count = 0

    @staticmethod
    def _save_checkpoint_snapshot(checkpoint_dir, agent, train_step_counter):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpointer = common.Checkpointer(
            ckpt_dir=str(checkpoint_dir),
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            global_step=train_step_counter
        )

        checkpointer.save(train_step_counter)

    @staticmethod
    def _copy_checkpoint_dir(src_dir, dst_dir):
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)

        if dst_dir.exists():
            shutil.rmtree(dst_dir)

        shutil.copytree(src_dir, dst_dir)

    @staticmethod
    def _build_checkpoint_selection_summary(eval_py_env, avg_return):
        records = eval_py_env.episode_records
        num_episodes = len(records)

        if num_episodes == 0:
            return {
                "num_episodes": 0,
                "avg_return": float(avg_return),
                "success_rate": 0.0,
                "collision_rate": 1.0,
                "obstacle_collision_rate": 1.0,
                "boundary_collision_rate": 1.0,
                "success_count": 0,
                "collision_count": 0
            }

        success_count = sum(
            int(record.get("success", 0))
            for record in records
        )

        collision_count = sum(
            int(record.get("collision", 0))
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
            "num_episodes": int(num_episodes),
            "avg_return": float(avg_return),
            "success_rate": success_count / num_episodes,
            "collision_rate": collision_count / num_episodes,
            "obstacle_collision_rate": obstacle_collision_count / num_episodes,
            "boundary_collision_rate": boundary_collision_count / num_episodes,
            "success_count": int(success_count),
            "collision_count": int(collision_count)
        }

    @staticmethod
    def _checkpoint_score(summary):
        """
        checkpoint 选择规则。

        优先级：
        1. success_rate 越高越好
        2. obstacle_collision_rate 越低越好
        3. boundary_collision_rate 越低越好
        4. collision_rate 越低越好
        5. avg_return 越高越好
        """

        return (
            float(summary["success_rate"]),
            -float(summary["obstacle_collision_rate"]),
            -float(summary["boundary_collision_rate"]),
            -float(summary["collision_rate"]),
            float(summary["avg_return"])
        )

    @staticmethod
    def _save_monitor_records(monitor_records, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "step",
            "train_loss",
            "avg_return",
            "early_collision_rate",
            "episode_count",
            "converged",
            "convergence_step",
            "convergence_return"
        ]

        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(monitor_records)

    @staticmethod
    def _save_checkpoint_selection_records(records, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "step",
            "train_loss",
            "avg_return",
            "success_rate",
            "collision_rate",
            "obstacle_collision_rate",
            "boundary_collision_rate",
            "success_count",
            "collision_count",
            "num_episodes",
            "checkpoint_dir",
            "is_best"
        ]

        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
