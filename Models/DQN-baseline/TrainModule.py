from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
from Models.TFEnv import *
from Models.Utils import *

class Train:
    def __init__(self):
        self.utils = Utils_stats()
    def train_tf_agents_dqn(self):
        # 超参数
        num_iterations = 200000 # 总训练迭代次数（不是回合数，是进行多少次梯度下降）
        initial_collect_steps = 5000 # 训练开始前，用随机动作收集多少步数据来“预热”经验池
        collect_steps_per_iteration = 1  # 每次梯度下降前，在环境中走几步收集新数据
        replay_buffer_max_length = 50000# 经验回放池的最大容量，满了会覆盖最旧的数据
        batch_size = 64  # 每次从经验池抓取多少条数据喂给神经网络
        learning_rate = 1e-4 # Adam 优化器的学习率
        log_interval = 200  # 每隔 200 步在控制台打印一次 Loss 损失值
        eval_interval = 2000  # 每隔 2000 步暂停训练，运行几局测试来评估当前模型的真实水平
        plot_steps = []
        plot_returns = []
        utils=Utils_stats()
        monitor_eval_episodes = 30
        monitor_records = []

        # 实例化写的环境包装器，并通过TFPyEnvironment在底层将python环境吐出的所有数据转换为TF原生的tf.Tensor张量，并自动加上一个batch维度表示第几个batch
        py_env=TFAgentSimulationEnv()
        tf_env=tf_py_environment.TFPyEnvironment(py_env)
        # 实例化评估专用环境
        eval_py_env=TFAgentSimulationEnv()
        eval_tf_env=tf_py_environment.TFPyEnvironment(eval_py_env)
        # 构建Q神经网络（6x12x3）输入层12维观测向量.隐藏层64，输出层3 维动作向量
        q_net = q_network.QNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=(64, 64)
        )
        # 优化器使用经典的Adam优化器，创建一个tf变量来记录当前是第几次迭代
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_step_counter=tf.Variable(0)
        # epsilon 是 DQN 中 ε-greedy 探索策略的随机动作概率。
        # epsilon=1.0 表示完全随机探索；epsilon=0.0 表示完全按照 Q 网络选择动作。
        # 为了解决训练初期 Q 网络尚未学习到有效策略的问题，初期使用较高 epsilon 进行充分探索。
        # 随着训练进行，逐渐降低 epsilon，让智能体更多依赖 Q 网络决策。
        # 这里借用 PolynomialDecay 来让 epsilon 从 1.0 衰减到 0.05。
        # 保留 0.05 的随机探索概率，可以降低策略过早固化的风险。
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.0,
            decay_steps=num_iterations,
            # 不为0是为了防止陷入局部最优解
            end_learning_rate=0.05
        )
        # 组装DQN Agent
        agent = dqn_agent.DqnAgent(
           tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_huber_loss,
            train_step_counter=train_step_counter,
            epsilon_greedy=lambda:epsilon_fn(train_step_counter),
            target_update_period = 500,
            gamma = 0.99
        )
        agent.initialize()

        run_dir = BASE_DIR / "runs" / "dqn_obs13_fc64_64_act5_5obs_(3)"
        checkpoint_dir = run_dir / "checkpoint"

        plot_path = run_dir / "Baseline_Convergence_Curve.png"
        collision_plot_path = run_dir / "Baseline_Early_Collision_Rate_Decay_Curve.png"
        episode_records_path = run_dir / "Baseline_Training_Episode_Records.csv"
        monitor_records_path = run_dir / "Baseline_Training_Monitor_Records.csv"
        run_dir.mkdir(parents=True, exist_ok=True)


        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            global_step=train_step_counter
        )

        # 构建经验回放缓冲区
        replay_buffer =tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_max_length
        )

        def collect_step(environment, policy, buffer):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj=trajectory.from_transition(time_step,action_step,next_time_step)
            buffer.add_batch(traj)

        print("正在进行初始化随机数据收集.....")
        random_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec()
        )
        for _ in range(initial_collect_steps):
            collect_step(tf_env, random_policy, replay_buffer)
        print("初始化收集完成，当前 replay buffer 帧数：", replay_buffer.num_frames().numpy())

        # 构建高性能数据管道
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2
        ).prefetch(3)

        iterator = iter(dataset)

        agent.train = common.function(agent.train)
        agent.train_step_counter.assign(0)

        for _ in range(num_iterations):
            # 1. 收集训练经验
            for _ in range(collect_steps_per_iteration):
                collect_step(tf_env, agent.collect_policy, replay_buffer)

            # 2. 训练一次 Q 网络
            experience, unused_info = next(iterator)
            loss_info = agent.train(experience)
            train_loss = loss_info.loss.numpy()

            # 3. 训练后读取 step
            step = agent.train_step_counter.numpy()

            # 4. 打印 loss
            if step % log_interval == 0:
                print(f"步数(step): {step} | 损失(loss): {train_loss:.4f}")

            # 5. 训练阶段监控指标
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
                    f"--- Baseline 训练监控 --- step: {step} | "
                    f"Avg Return: {avg_return:.2f} | "
                    f"Early Collision Rate: {early_collision_rate:.3f} | "
                    f"Episode Count: {py_env.episode_count}"
                )

                train_checkpointer.save(train_step_counter)

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
        print("\n[Baseline DQN] 训练结束，保存最终 checkpoint...")
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

        print(f"Baseline Training Early Collision Rate: {final_early_collision_rate:.3f}")
        print(f"Baseline Training Success Rate: {training_success_rate:.3f}")
        print(f"Baseline Convergence Stats: {final_convergence_stats}")

        self.utils.save_episode_records(
            py_env,
            episode_records_path
        )

        self._save_monitor_records(
            monitor_records,
            monitor_records_path
        )

        print(f"Baseline episode records 已保存至: {episode_records_path}")
        print(f"Baseline monitor records 已保存至: {monitor_records_path}")
        print(f"Baseline convergence curve 已保存至: {plot_path}")
        print(f"Baseline collision curve 已保存至: {collision_plot_path}")

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

