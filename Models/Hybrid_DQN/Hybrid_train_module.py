import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.utils import common

try:
    from Models.TFEnv import TFAgentSimulationEnv
    from Models.Utils import Utils_stats
except ModuleNotFoundError:
    from Models.TFEnv import TFAgentSimulationEnv
    from Models.Utils import Utils_stats


BASE_DIR = Path(__file__).resolve().parent


class GAHeuristicController:
    """
    GA 进化出的低维启发式控制器。

    输入:
        state: 13 维状态
            state[:11] = 11 根射线距离，范围 [0, 1]
            state[11] = normalized_yaw，范围 [-1, 1]
            state[12] = normalized_target_dist，范围 [0, 1]

    输出:
        action_index: 0..4

    动作映射由 TFAgentSimulationEnv 负责：
        0 -> -1.0
        1 -> -0.5
        2 ->  0.0
        3 ->  0.5
        4 ->  1.0
    """

    def __init__(self, params_path):
        self.params_path = Path(params_path)

        if not self.params_path.exists():
            raise FileNotFoundError(
                f"找不到 GA heuristic 参数文件: {self.params_path}\n"
                f"请先运行 GA_heuristic.py 生成 best_ga_heuristic_params.npy"
            )

        self.params = np.load(self.params_path).astype(np.float32)

        if self.params.shape[0] != 7:
            raise ValueError(
                f"GA heuristic 参数维度错误: expected 7, got {self.params.shape[0]}"
            )

        print(f"[GA-Heuristic] 已加载参数: {self.params_path}")
        print(f"[GA-Heuristic] params = {self.params}")

    def action(self, state):
        state = np.asarray(state, dtype=np.float32)

        rays = np.clip(state[:11], 0.0, 1.0)
        yaw = float(np.clip(state[11], -1.0, 1.0))
        target_dist = float(np.clip(state[12], 0.0, 1.0))

        w_target = self.params[0]
        w_side = self.params[1]
        w_front = self.params[2]
        w_front_group = self.params[3]
        bias = self.params[4]
        threshold_small = abs(float(self.params[5]))
        threshold_large = abs(float(self.params[6]))

        if threshold_large <= threshold_small:
            threshold_large = threshold_small + 0.1

        left_rays = rays[:5]
        front_ray = rays[5]
        right_rays = rays[6:]

        left_danger = float(np.mean(1.0 - left_rays))
        right_danger = float(np.mean(1.0 - right_rays))
        front_danger = float(1.0 - front_ray)
        front_group_danger = float(np.mean(1.0 - rays[4:7]))

        side_balance = left_danger - right_danger

        if abs(side_balance) > 1e-6:
            escape_sign = np.sign(side_balance)
        else:
            escape_sign = np.sign(yaw) if abs(yaw) > 1e-6 else 0.0

        target_term = w_target * yaw
        side_term = w_side * side_balance
        front_term = w_front * front_danger * escape_sign
        front_group_term = w_front_group * front_group_danger * escape_sign

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


class Hybrid_Train:
    """
    Hybrid GA-Heuristic-DQN 训练模块。

    职责:
        1. 使用 GA-Heuristic policy 收集 initial replay buffer；
        2. 使用 DQN 继续训练；
        3. 记录训练阶段 Average Return over Training Steps；
        4. 记录 Early Collision Rate 和 Collision Rate Decay Curve；
        5. 保存 checkpoint 和 episode records。

    注意:
        这里不再做 GA 权重注入。
        GA 只负责 warm-start replay buffer。
    """

    def __init__(self):
        self.utils = Utils_stats()

    def train_hybrid_agent(self):
        # -----------------------------
        # 超参数
        # -----------------------------
        num_iterations = 200000
        initial_collect_steps = 5000
        collect_steps_per_iteration = 1

        replay_buffer_max_length = 50000
        batch_size = 64
        learning_rate = 1e-4

        log_interval = 200
        eval_interval = 2000
        monitor_eval_episodes = 30

        fc_layer_params = (64, 64)

        # Hybrid 的 DQN 探索率可以低于 baseline，但不建议太低。
        # 因为 QNetwork 仍然是随机初始化，只是 replay buffer 更好。
        epsilon_initial = 0.5
        epsilon_final = 0.05

        run_dir = BASE_DIR / "runs" / "hybrid_ga_heuristic_dqn_obs13_fc64_64_act5_5obs(3)"
        checkpoint_dir = run_dir / "checkpoint"

        plot_path = run_dir / "Hybrid_Convergence_Curve.png"
        collision_plot_path = run_dir / "Hybrid_Early_Collision_Rate_Decay_Curve.png"
        episode_records_path = run_dir / "Hybrid_Training_Episode_Records.csv"
        monitor_records_path = run_dir / "Hybrid_Training_Monitor_Records.csv"

        ga_params_path = (
            BASE_DIR
            / "runs"
            / "ga_heuristic_controller_5obs"
            / "best_ga_heuristic_params_5obs.npy"
        )

        run_dir.mkdir(parents=True, exist_ok=True)

        plot_steps = []
        plot_returns = []
        monitor_records = []

        # -----------------------------
        # 环境
        # -----------------------------
        py_env = TFAgentSimulationEnv()
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

        eval_py_env = TFAgentSimulationEnv()
        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        # -----------------------------
        # GA-Heuristic Controller
        # -----------------------------
        ga_controller = GAHeuristicController(ga_params_path)

        # -----------------------------
        # DQN Agent
        # -----------------------------
        q_net = q_network.QNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=fc_layer_params
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        train_step_counter = tf.Variable(0)

        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=epsilon_initial,
            decay_steps=int(num_iterations * 0.8),
            end_learning_rate=epsilon_final
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

        train_checkpointer = common.Checkpointer(
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

        # -----------------------------
        # 收集函数
        # -----------------------------
        def collect_dqn_step(environment, policy, buffer):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)

            traj = trajectory.from_transition(
                time_step,
                action_step,
                next_time_step
            )

            buffer.add_batch(traj)

        def collect_ga_heuristic_step(environment, controller, buffer):
            """
            使用 GA-Heuristic controller 收集一条 transition。
            这是 Hybrid 相比 Baseline 的核心差异。
            """

            time_step = environment.current_time_step()

            observation = time_step.observation.numpy()[0]
            action_index = controller.action(observation)

            action_tensor = tf.constant(
                [action_index],
                dtype=tf.int32
            )

            action_step = policy_step.PolicyStep(
                action=action_tensor,
                state=(),
                info=()
            )

            next_time_step = environment.step(action_tensor)

            traj = trajectory.from_transition(
                time_step,
                action_step,
                next_time_step
            )

            buffer.add_batch(traj)

        # -----------------------------
        # GA-Heuristic Warm-start
        # -----------------------------
        print("\n[Hybrid Warm-start] 使用 GA-Heuristic policy 收集初始经验...")
        print(f"[Hybrid Warm-start] initial_collect_steps = {initial_collect_steps}")

        for i in range(initial_collect_steps):
            collect_ga_heuristic_step(
                tf_env,
                ga_controller,
                replay_buffer
            )

            if (i + 1) % 1000 == 0:
                print(
                    f"[Hybrid Warm-start] 已收集 {i + 1}/{initial_collect_steps} steps | "
                    f"当前训练环境 episode_count = {py_env.episode_count}"
                )

        print(
            "[Hybrid Warm-start] 初始化收集完成，当前 replay buffer 帧数:",
            replay_buffer.num_frames().numpy()
        )

        warm_start_early_collision_rate = self.utils.compute_early_collision_rate(
            py_env,
            first_n_episodes=500
        )

        print(
            f"[Hybrid Warm-start] Early Collision Rate over first 500 episodes: "
            f"{warm_start_early_collision_rate:.3f}"
        )

        # -----------------------------
        # Dataset
        # -----------------------------
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2
        ).prefetch(3)

        iterator = iter(dataset)

        agent.train = common.function(agent.train)
        agent.train_step_counter.assign(0)

        # -----------------------------
        # DQN 训练循环
        # -----------------------------
        print("\n[Hybrid DQN] 开始训练...")

        for _ in range(num_iterations):
            # 1. 使用 DQN collect_policy 继续收集经验
            for _ in range(collect_steps_per_iteration):
                collect_dqn_step(
                    tf_env,
                    agent.collect_policy,
                    replay_buffer
                )

            # 2. 从 replay buffer 采样并训练
            experience, unused_info = next(iterator)
            loss_info = agent.train(experience)
            train_loss = float(loss_info.loss.numpy())

            step = int(agent.train_step_counter.numpy())

            # 3. 打印 loss
            if step % log_interval == 0:
                print(
                    f"步数(step): {step} | "
                    f"损失(loss): {train_loss:.4f} | "
                    f"训练环境episode数: {py_env.episode_count}"
                )

            # 4. 训练阶段监控
            if step % eval_interval == 0:
                avg_return = self.utils.compute_avg_return(
                    eval_tf_env,
                    agent.policy,
                    num_episodes=monitor_eval_episodes
                )

                early_collision_rate = self.utils.compute_early_collision_rate(
                    py_env,
                    first_n_episodes=500
                )

                convergence_stats = self._compute_convergence_steps(
                    plot_steps=plot_steps,
                    plot_returns=plot_returns,
                    threshold=25.0,
                    window_size=5
                )

                plot_steps.append(step)
                plot_returns.append(avg_return)

                monitor_records.append({
                    "step": step,
                    "train_loss": train_loss,
                    "avg_return": avg_return,
                    "early_collision_rate": early_collision_rate,
                    "episode_count": py_env.episode_count,
                    "converged": int(convergence_stats["converged"]),
                    "convergence_step": convergence_stats["convergence_step"],
                    "convergence_return": convergence_stats["convergence_return"]
                })

                print(
                    f"--- Hybrid 训练监控 --- step: {step} | "
                    f"Avg Return: {avg_return:.2f} | "
                    f"Early Collision Rate: {early_collision_rate:.3f} | "
                    f"Episode Count: {py_env.episode_count}"
                )

                train_checkpointer.save(train_step_counter)

                self.utils.plot_convergence_curve(
                    plot_steps=plot_steps,
                    plot_returns=plot_returns,
                    plot_path=plot_path,
                    label="Hybrid DQN"
                )

                self.utils.plot_collision_rate_curve(
                    py_env=py_env,
                    save_path=collision_plot_path,
                    first_n_episodes=500,
                    window_size=50,
                    title="Hybrid GA-Heuristic Early Collision Rate Decay Curve",
                )

        # -----------------------------
        # 训练结束
        # -----------------------------
        print("\n[Hybrid DQN] 训练结束，保存最终 checkpoint...")
        train_checkpointer.save(train_step_counter)
        print(f"最终 checkpoint 已保存至: {checkpoint_dir}")

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

        print(f"Hybrid Training Early Collision Rate: {final_early_collision_rate:.3f}")
        print(f"Hybrid Training Success Rate: {training_success_rate:.3f}")
        print(f"Hybrid Convergence Stats: {final_convergence_stats}")

        self.utils.save_episode_records(
            py_env,
            episode_records_path
        )

        self._save_monitor_records(
            monitor_records,
            monitor_records_path
        )

        print(f"Hybrid episode records 已保存至: {episode_records_path}")
        print(f"Hybrid monitor records 已保存至: {monitor_records_path}")
        print(f"Hybrid convergence curve 已保存至: {plot_path}")
        print(f"Hybrid collision curve 已保存至: {collision_plot_path}")

        return {
            "run_dir": run_dir,
            "checkpoint_dir": checkpoint_dir,
            "episode_records_path": episode_records_path,
            "monitor_records_path": monitor_records_path,
            "final_step": int(agent.train_step_counter.numpy()),
            "early_collision_rate": final_early_collision_rate,
            "training_success_rate": training_success_rate,
            "convergence_stats": final_convergence_stats
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


