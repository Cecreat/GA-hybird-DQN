import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


from DQN_agent import TFAgentSimulationEnv, compute_avg_return


def inject_ga_weights(agent, weights_path):
    print(f"\n[记忆移植] 正在从 {weights_path} 读取 GA 先天生存本能...")
    import numpy as np

    try:
        weights_1d = np.load(weights_path)
    except FileNotFoundError:
        print(f"找不到权重文件 {weights_path}，请先运行 GA.py！")
        return

    # 【注意】这里必须和您设置的网络维度严格一致！
    input_dim = 6
    hidden_dim1 = 32
    hidden_dim2 = 32
    output_dim = 3

    # 计算切片索引 (与 GA.py 完全一致)
    idx1 = input_dim * hidden_dim1
    idx2 = idx1 + hidden_dim1
    idx3 = idx2 + hidden_dim1 * hidden_dim2
    idx4 = idx3 + hidden_dim2
    idx5 = idx4 + hidden_dim2 * output_dim

    # 切片并重塑为矩阵
    W1 = weights_1d[0:idx1].reshape((input_dim, hidden_dim1))
    b1 = weights_1d[idx1:idx2]

    W2 = weights_1d[idx2:idx3].reshape((hidden_dim1, hidden_dim2))
    b2 = weights_1d[idx3:idx4]

    W3 = weights_1d[idx4:idx5].reshape((hidden_dim2, output_dim))
    b3 = weights_1d[idx5:]

    # 组合成 TensorFlow 期望的 6 个权重的列表顺序
    weights_list = [W1, b1, W2, b2, W3, b3]

    try:
        # TF-Agents 的 QNetwork 在未调用过前无法获知形状
        # 强制创建一次虚拟输入来初始化网络权重结构
        dummy_state = tf.zeros((1, input_dim))
        agent._q_network(dummy_state)

        agent._q_network.set_weights(weights_list)
        print("[记忆移植成功] GA 优良基因已完全注入 RL 神经网络！")
    except Exception as e:
        print(f"[记忆移植失败] 权重形状不匹配: {e}")


def train_hybrid_agent():

    num_iterations = 60000
    initial_collect_steps = 1000
    batch_size = 64
    learning_rate = 1e-4
    log_interval = 200
    eval_interval = 1000

    py_env = TFAgentSimulationEnv()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_py_env = TFAgentSimulationEnv()
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    plot_steps = []
    plot_returns = []

    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(32, 32)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.15,
        decay_steps=int(num_iterations * 0.8),
        end_learning_rate=0.05
    )


    agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter)
    )


    agent.initialize()

    checkpoint_dir = os.path.join(os.getcwd(), '../hybrid_checkpoint')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        global_step=train_step_counter
    )

    inject_ga_weights(agent, "D:\MSc_Project/best_ga_weights.npy")


    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=10000
    )

    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)


    print("正在使用遗传基线策略收集高质量预热数据 (避免纯随机撞墙)...")

    for _ in range(initial_collect_steps):
        collect_step(tf_env, agent.collect_policy, replay_buffer)


    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    print("\n=========================================")
    print("混合进化强化学习 (Hybrid GA-RL) 训练正式开始！")
    print("=========================================")


    for _ in range(num_iterations):
        collect_step(tf_env, agent.collect_policy, replay_buffer)
        experience, unused_info = next(iterator)
        loss_info = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print(f"步数(Step): {step:5d} | 损失(Loss): {loss_info.loss.numpy():.4f}")

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_tf_env, agent.policy, num_episodes=10)
            print(f"--- 评估 --- 步数: {step:5d} | 平均回报 (Avg Return): {avg_return:.2f}")
            plot_steps.append(step)
            plot_returns.append(avg_return)
            train_checkpointer.save(train_step_counter)

            plt.figure(figsize=(10, 6))
            plt.plot(plot_steps, plot_returns, label='Hybrid GA-RL', color='red', linewidth=2)
            plt.xlabel('Training Steps')
            plt.ylabel('Average Return')
            plt.title('Convergence Speed: Evaluation Return over Training Steps')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()


            plt.savefig('hybrid_convergence_curve.png', dpi=300)
            print("\n收敛曲线已保存为: hybrid_convergence_curve.png")

    from tf_agents.policies import policy_saver
    save_dir = os.path.join(os.getcwd(), '../hybrid_ga_rl_policy')
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save(save_dir)
    print(f"\n混合模型已成功保存至: {save_dir}")


if __name__ == "__main__":
    train_hybrid_agent()