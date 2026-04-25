import os
import numpy as np
import tensorflow as tf


from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


from DQN_agent import TFAgentSimulationEnv, compute_avg_return


def inject_ga_weights(agent, npy_path="best_ga_weights.npy"):

    print(f"\n[记忆移植] 正在从 {npy_path} 读取 GA 先天生存本能...")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"未找到 {npy_path}！请先运行 GA.py 生成最优基线权重。")


    flat_weights = np.load(npy_path)


    w1 = flat_weights[0:72].reshape((6, 12)).astype(np.float32)
    b1 = flat_weights[72:84].astype(np.float32)
    w2 = flat_weights[84:120].reshape((12, 3)).astype(np.float32)
    b2 = flat_weights[120:123].astype(np.float32)

    weights_list = [w1, b1, w2, b2]

    # 4. 暴力覆盖主网络 (Q-Network) 和目标网络 (Target-Network)
    agent._q_network.set_weights(weights_list)
    agent._target_q_network.set_weights(weights_list)

    print("[记忆移植] 注入成功！智能体已摆脱冷启动，赢在起跑线！\n")


def train_hybrid_agent():

    num_iterations = 20000
    initial_collect_steps = 1000
    batch_size = 64
    learning_rate = 1e-3
    log_interval = 200
    eval_interval = 1000

    py_env = TFAgentSimulationEnv()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_py_env = TFAgentSimulationEnv()
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(12,)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.2,
        decay_steps=num_iterations // 2,
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


    inject_ga_weights(agent, "best_ga_weights.npy")


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
    print("🚀 混合进化强化学习 (Hybrid GA-RL) 训练正式开始！")
    print("=========================================")


    for _ in range(num_iterations):
        collect_step(tf_env, agent.collect_policy, replay_buffer)
        experience, unused_info = next(iterator)
        loss_info = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print(f"步数(Step): {step:5d} | 损失(Loss): {loss_info.loss.numpy():.4f}")

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_tf_env, agent.policy, num_episodes=5)
            print(f"🌟 --- 评估 --- 步数: {step:5d} | 平均回报 (Avg Return): {avg_return:.2f}")


    from tf_agents.policies import policy_saver
    save_dir = os.path.join(os.getcwd(), 'hybrid_ga_rl_policy')
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save(save_dir)
    print(f"\n🎉 混合模型已成功保存至: {save_dir}")


if __name__ == "__main__":
    train_hybrid_agent()