import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.utils import common

try:
    from Models.TFEnv import TFAgentSimulationEnv
    from Models.Utils import Utils_stats
except ModuleNotFoundError:
    from Models.TFEnv import TFAgentSimulationEnv
    from Models.Utils import Utils_stats


BASE_DIR = Path(__file__).resolve().parent

SINGLE_FRAME_OBS_DIM = 13
FRAME_STACK = 4
STACKED_OBS_DIM = SINGLE_FRAME_OBS_DIM * FRAME_STACK


def sigmoid_epsilon(step,num_iterations,epsilon_initial,epsilon_final):
    step=tf.cast(step, tf.float32)
    decay_steps = tf.cast(int(num_iterations * 0.8), tf.float32)
    step = tf.clip_by_value(step, 0.0, decay_steps)

    epsilon_start = tf.constant(epsilon_initial, dtype=tf.float32)
    epsilon_end = tf.constant(epsilon_final, dtype=tf.float32)

    tau = decay_steps / 10.0
    x = (decay_steps / 2.0 - step) / tau
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * tf.math.sigmoid(x)
    return epsilon


class GAHeuristicController:
    """
    GA 进化出的低维启发式控制器。

    输入:
        state: 13 维单帧状态，或 52 维 frame_stack=4 状态。
        如果是 52 维 stacked observation，控制器只取最后 13 维作为当前帧。

        单帧结构:
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


    @staticmethod
    def _extract_latest_observation(state):
        """
        兼容单帧 13 维 observation 和 frame-stacked observation。
        Hybrid DQN 使用 4 帧堆叠时，DQN 接收完整 52 维输入；
        GA heuristic controller 仍只读取最新一帧，避免扩大 GA 参数搜索空间。
        """
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        if state.shape[0] == SINGLE_FRAME_OBS_DIM:
            return state

        if state.shape[0] > SINGLE_FRAME_OBS_DIM and state.shape[0] % SINGLE_FRAME_OBS_DIM == 0:
            return state[-SINGLE_FRAME_OBS_DIM:]

        raise ValueError(
            f"无法解析 observation 维度: {state.shape[0]}。"
            f"期望 {SINGLE_FRAME_OBS_DIM} 或其整数倍。"
        )

    def action(self, state):
        state = self._extract_latest_observation(state)

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
        5. 保存 latest checkpoint、candidate checkpoints 和 best checkpoint。

    注意:
        GA 只负责筛选成功轨迹用于 warm-start replay buffer，不做网络权重注入。
    """

    def __init__(self):
        self.utils = Utils_stats()

    def train_hybrid_agent(self):
        # -----------------------------
        # 超参数
        # -----------------------------
        num_iterations = 200000
        initial_collect_steps = 5000
        ga_success_collect_target_steps = initial_collect_steps
        ga_max_warm_start_episodes = 2000
        ga_max_steps_per_episode = 500
        ga_max_path_efficiency_ratio = 2.0
        random_fill_if_ga_insufficient = True
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

        fc_layer_params = (128, 128)

        epsilon_initial = 0.5
        epsilon_final = 0.01

        run_dir = BASE_DIR / "runs" / "hybrid_ga_heuristic_dqn_framestack4_obs52_fc64_64_act5_15obstacles_improve_epsilon_0.01"
        checkpoint_dir = run_dir / "checkpoint"                 # latest checkpoint
        candidate_checkpoint_root = run_dir / "candidate_checkpoints"
        best_checkpoint_dir = run_dir / "best_checkpoint"       # final evaluation 使用这个

        plot_path = run_dir / "Hybrid_Convergence_Curve.png"
        collision_plot_path = run_dir / "Hybrid_Early_Collision_Rate_Decay_Curve.png"
        episode_records_path = run_dir / "Hybrid_Training_Episode_Records.csv"
        monitor_records_path = run_dir / "Hybrid_Training_Monitor_Records.csv"
        checkpoint_selection_records_path = run_dir / "Hybrid_Checkpoint_Selection_Records.csv"
        warm_start_records_path = run_dir / "Hybrid_GA_Success_WarmStart_Records.csv"

        ga_params_candidates = [
            BASE_DIR / "runs" / "ga_heuristic_controller_15obs_improve" / "best_ga_heuristic_params_15obs_improve.npy",
        ]
        ga_params_path = self._resolve_existing_path(ga_params_candidates)

        run_dir.mkdir(parents=True, exist_ok=True)
        self._clear_dir(candidate_checkpoint_root)
        self._clear_dir(best_checkpoint_dir)

        plot_steps = []
        plot_returns = []
        monitor_records = []
        checkpoint_selection_records = []
        warm_start_records = []
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
            epsilon_greedy=lambda: sigmoid_epsilon(train_step_counter,num_iterations,epsilon_initial,epsilon_final),
            target_update_period=500,
            gamma=0.99
        )

        agent.initialize()

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

        def collect_ga_heuristic_episode(environment, py_environment, controller, max_steps_per_episode):
            """
            使用 GA-Heuristic controller 收集一个完整 episode。

            关键区别：
                这里只返回 episode 的 transition 列表，不会立即写入 replay buffer。
                是否写入由 episode 结束后的质量筛选结果决定。
            """

            time_step = environment.reset()
            episode_trajs = []
            episode_return = 0.0
            step_count = 0
            terminated_by_env = False
            final_info = {}

            while step_count < max_steps_per_episode:
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

                episode_trajs.append(traj)
                episode_return += float(next_time_step.reward.numpy()[0])
                step_count += 1

                if bool(next_time_step.is_last().numpy()[0]):
                    terminated_by_env = True
                    final_info = dict(getattr(py_environment, "last_info", {}))
                    break

                time_step = next_time_step

            if terminated_by_env:
                termination_reason = final_info.get("termination_reason", "unknown")
                collision = bool(final_info.get("collision", False))
                success = bool(final_info.get("success", False))
                path_efficiency_ratio = final_info.get("path_efficiency_ratio", None)
            else:
                termination_reason = "timeout"
                collision = False
                success = False
                path_efficiency_ratio = None

            return {
                "trajectories": episode_trajs,
                "num_transitions": len(episode_trajs),
                "steps": step_count,
                "episode_return": episode_return,
                "termination_reason": termination_reason,
                "collision": collision,
                "success": success,
                "path_efficiency_ratio": path_efficiency_ratio,
                "accepted": False,
                "reject_reason": None,
            }

        # -----------------------------
        # GA-Heuristic Success-Trajectory Warm-start
        # -----------------------------
        print("\n[Hybrid Warm-start] 使用 GA-Heuristic 生成并筛选成功轨迹...")
        print(f"[Hybrid Warm-start] GA accepted transition target = {ga_success_collect_target_steps}")
        print(f"[Hybrid Warm-start] max GA warm-start episodes = {ga_max_warm_start_episodes}")
        print(f"[Hybrid Warm-start] max path efficiency ratio = {ga_max_path_efficiency_ratio}")

        accepted_ga_steps = 0
        accepted_ga_episodes = 0
        generated_ga_episodes = 0
        rejected_collision_episodes = 0
        rejected_timeout_episodes = 0
        rejected_low_quality_success_episodes = 0

        while (
            accepted_ga_steps < ga_success_collect_target_steps
            and generated_ga_episodes < ga_max_warm_start_episodes
        ):
            generated_ga_episodes += 1

            episode_result = collect_ga_heuristic_episode(
                environment=tf_env,
                py_environment=py_env,
                controller=ga_controller,
                max_steps_per_episode=ga_max_steps_per_episode
            )

            accepted, reject_reason = self._is_high_quality_ga_episode(
                episode_result,
                max_path_efficiency_ratio=ga_max_path_efficiency_ratio
            )

            episode_result["accepted"] = accepted
            episode_result["reject_reason"] = reject_reason

            if accepted:
                self._add_episode_trajectories_to_buffer(
                    replay_buffer,
                    episode_result["trajectories"]
                )
                accepted_ga_steps += episode_result["num_transitions"]
                accepted_ga_episodes += 1
            else:
                if episode_result["termination_reason"] in ["obstacle_collision", "boundary_collision"]:
                    rejected_collision_episodes += 1
                elif episode_result["termination_reason"] == "timeout":
                    rejected_timeout_episodes += 1
                elif reject_reason == "path_efficiency_too_low":
                    rejected_low_quality_success_episodes += 1

            warm_start_records.append({
                "ga_episode": generated_ga_episodes,
                "accepted": int(accepted),
                "reject_reason": reject_reason,
                "termination_reason": episode_result["termination_reason"],
                "collision": int(episode_result["collision"]),
                "success": int(episode_result["success"]),
                "steps": int(episode_result["steps"]),
                "num_transitions": int(episode_result["num_transitions"]),
                "episode_return": float(episode_result["episode_return"]),
                "path_efficiency_ratio": episode_result["path_efficiency_ratio"],
                "accepted_ga_steps_so_far": int(accepted_ga_steps),
                "replay_buffer_frames": int(replay_buffer.num_frames().numpy())
            })

            if generated_ga_episodes % 50 == 0 or accepted:
                print(
                    f"[Hybrid Warm-start] generated_ep={generated_ga_episodes} | "
                    f"accepted_ep={accepted_ga_episodes} | "
                    f"accepted_steps={accepted_ga_steps}/{ga_success_collect_target_steps} | "
                    f"last_reason={episode_result['termination_reason']} | "
                    f"accepted={int(accepted)}"
                )

        random_fill_steps = 0
        replay_frames = int(replay_buffer.num_frames().numpy())

        if random_fill_if_ga_insufficient and replay_frames < initial_collect_steps:
            random_fill_steps = initial_collect_steps - replay_frames
            print(
                f"[Hybrid Warm-start] GA 成功轨迹不足，使用 random policy 补齐 replay buffer: "
                f"{random_fill_steps} steps"
            )

            random_policy = random_tf_policy.RandomTFPolicy(
                tf_env.time_step_spec(),
                tf_env.action_spec()
            )

            for i in range(random_fill_steps):
                collect_dqn_step(
                    tf_env,
                    random_policy,
                    replay_buffer
                )

                if (i + 1) % 1000 == 0:
                    print(
                        f"[Hybrid Warm-start Random Fill] 已补齐 {i + 1}/{random_fill_steps} steps | "
                        f"当前 replay buffer 帧数 = {replay_buffer.num_frames().numpy()}"
                    )

        print(
            "[Hybrid Warm-start] 初始化收集完成，当前 replay buffer 帧数:",
            replay_buffer.num_frames().numpy()
        )

        warm_start_total_generated = max(generated_ga_episodes, 1)
        warm_start_ga_acceptance_rate = accepted_ga_episodes / warm_start_total_generated
        warm_start_ga_collision_rate = rejected_collision_episodes / warm_start_total_generated
        warm_start_ga_timeout_rate = rejected_timeout_episodes / warm_start_total_generated

        warm_start_early_collision_rate = self.utils.compute_early_collision_rate(
            py_env,
            first_n_episodes=500
        )

        print(
            f"[Hybrid Warm-start Summary] generated_ga_episodes={generated_ga_episodes} | "
            f"accepted_ga_episodes={accepted_ga_episodes} | "
            f"accepted_ga_steps={accepted_ga_steps} | "
            f"random_fill_steps={random_fill_steps}"
        )
        print(
            f"[Hybrid Warm-start Summary] GA acceptance rate={warm_start_ga_acceptance_rate:.3f} | "
            f"GA collision reject rate={warm_start_ga_collision_rate:.3f} | "
            f"GA timeout reject rate={warm_start_ga_timeout_rate:.3f} | "
            f"training early collision rate={warm_start_early_collision_rate:.3f}"
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

            # 4. 训练阶段监控 + best checkpoint selection
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
                    f"--- Hybrid 训练监控 --- step: {step} | "
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
                        f"[Hybrid Best Checkpoint Updated] step={step} | "
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
        print("\n[Hybrid DQN] 训练结束，保存最终 latest checkpoint...")
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

        print(f"Hybrid Training Early Collision Rate: {final_early_collision_rate:.3f}")
        print(f"Hybrid Training Success Rate: {training_success_rate:.3f}")
        print(f"Hybrid Convergence Stats: {final_convergence_stats}")
        print(f"Hybrid Best Checkpoint Dir: {best_checkpoint_dir}")
        print(f"Hybrid Best Selection Record: {best_selection_record}")

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

        self._save_warm_start_records(
            warm_start_records,
            warm_start_records_path
        )

        print(f"Hybrid episode records 已保存至: {episode_records_path}")
        print(f"Hybrid monitor records 已保存至: {monitor_records_path}")
        print(f"Hybrid checkpoint selection records 已保存至: {checkpoint_selection_records_path}")
        print(f"Hybrid GA success warm-start records 已保存至: {warm_start_records_path}")
        print(f"Hybrid convergence curve 已保存至: {plot_path}")
        print(f"Hybrid collision curve 已保存至: {collision_plot_path}")

        return {
            "run_dir": run_dir,
            "checkpoint_dir": checkpoint_dir,
            "best_checkpoint_dir": best_checkpoint_dir,
            "candidate_checkpoint_root": candidate_checkpoint_root,
            "episode_records_path": episode_records_path,
            "monitor_records_path": monitor_records_path,
            "checkpoint_selection_records_path": checkpoint_selection_records_path,
            "warm_start_records_path": warm_start_records_path,
            "final_step": int(agent.train_step_counter.numpy()),
            "early_collision_rate": final_early_collision_rate,
            "training_success_rate": training_success_rate,
            "convergence_stats": final_convergence_stats,
            "best_selection_record": best_selection_record,
            "warm_start_early_collision_rate": warm_start_early_collision_rate,
            "accepted_ga_episodes": accepted_ga_episodes,
            "accepted_ga_steps": accepted_ga_steps,
            "generated_ga_episodes": generated_ga_episodes,
            "random_fill_steps": random_fill_steps
        }

    @staticmethod
    def _resolve_existing_path(candidate_paths):
        for path in candidate_paths:
            path = Path(path)
            if path.exists():
                print(f"[GA-Heuristic] 使用参数文件: {path}")
                return path

        candidate_text = "\n".join(str(Path(p)) for p in candidate_paths)
        raise FileNotFoundError(
            "找不到 GA heuristic 参数文件。已检查以下路径:\n"
            f"{candidate_text}\n"
            "请先运行 GA.py，或手动修改 ga_params_candidates。"
        )

    @staticmethod
    def _is_high_quality_ga_episode(episode_result, max_path_efficiency_ratio=None):
        """
        GA warm-start 质量筛选规则。

        当前策略：
            1. 必须 success；
            2. 如果设置了 max_path_efficiency_ratio，则过度绕路的成功轨迹也丢弃。
        """

        if not bool(episode_result.get("success", False)):
            return False, "not_success"

        path_efficiency_ratio = episode_result.get("path_efficiency_ratio", None)

        if (
            max_path_efficiency_ratio is not None
            and path_efficiency_ratio is not None
            and float(path_efficiency_ratio) > float(max_path_efficiency_ratio)
        ):
            return False, "path_efficiency_too_low"

        return True, "accepted_success_trajectory"

    @staticmethod
    def _add_episode_trajectories_to_buffer(buffer, episode_trajectories):
        for traj in episode_trajectories:
            buffer.add_batch(traj)

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

    @staticmethod
    def _save_warm_start_records(records, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "ga_episode",
            "accepted",
            "reject_reason",
            "termination_reason",
            "collision",
            "success",
            "steps",
            "num_transitions",
            "episode_return",
            "path_efficiency_ratio",
            "accepted_ga_steps_so_far",
            "replay_buffer_frames"
        ]

        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

