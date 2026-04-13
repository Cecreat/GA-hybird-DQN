import numpy as np
import pygame
import tensorflow as tf
from tensorflow.python.ops.nn_impl_distribute import compute_average_loss
# from tensorflow_probability import optimizer

# TF-Agents 相关组件
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

# 导入我们自己写的 Pygame 环境
from Env import SimulationEnv

# 将python环境包装成TF环境
class TFAgentSimulationEnv(py_environment.PyEnvironment):
    def __init__(self):
        # 调用父类构造函数初始化基础环境所需的变量
        super().__init__()
        # 实例化写好的python环境
        self._env = SimulationEnv()
        # TF在处理离散动作的时候默认从0开始正整数索引，所以需要通过映射解决“语言不同”的问题
        self.action_mapping={0:-1,1:0,2:1}
        # 定义动作空间的规格：  shape=()表明是标量，dtype=np.int32表明必须是32位整数，minimum=0,maximum=2明确范围 本质是神经网络的输出空间
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),dtype=np.int32,minimum=0,maximum=2,name='action'
        )
        # 定义观察空间的规格  shape=(6,)表明是一个包含六个元素的一维数组 本质是神经网络的输入空间
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,),dtype=np.float32,minimum=-1.0, maximum=1.0, name='observation'
        )
        # 初始化内部状态占位符维一个全为0的数组，并设置回合结束标志位为False
        self._state=np.zeros((6,),dtype=np.float32)
        self._episode_ended = False
    # 获取两个规格的接口
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    # 触发底层真实的物理环境进行重置
    def _reset(self):
        # 调用python环境中的reset重置环境并获取新的状态（6维向量，5个射线的感知结果和1个与目标点的角度差）
        state=self._env.reset()
        # 将我们python环境中的状态列表转化为Numpy的float32数组，严格匹配定义的观察空间
        self._state=np.array(state,dtype=np.float32)
        # 重置回合结束标志
        self._episode_ended = False
        # TF的智能体不接收裸状态的数据，需要将数据包裹在ts.restart对象中，ts.restart会生成一个特殊的信号，告诉经验回放池从这条数据开始是一个全新回合的起点
        return ts.restart(self._state)
    # TF-Agent在训练循环有时会在回合结束后多执行一步。如果检测到当前回合已经结束，强制调用self.reset来自动开启下一轮，防止物理环境抛出异常
    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        # 将网络传进来的正整数动作序列映射为物理环境中需要的-1，0，1
        real_action=self.action_mapping[int(action)]
        # 获取环境返回的新状态、奖励值、是否撞毁/胜利
        next_state,reward,done=self._env.step(real_action)
        # next_state转化为符合规格的float32数组
        self._state=np.array(next_state,dtype=np.float32)
        if done:
            self._episode_ended=True
            # 告诉Q网络吗，这是最后一步了，没有未来的预期收益了，折扣因子应当按照0计算
            return ts.termination(self._state,reward=reward)
        else:
            return ts.transition(self._state,reward=reward,discount=0.99)
def train_tf_agents_dqn():
    # 超参数
    num_iterations = 20000  # 总训练迭代次数（不是回合数，是进行多少次梯度下降）
    initial_collect_steps = 1000  # 训练开始前，用随机动作收集多少步数据来“预热”经验池
    collect_steps_per_iteration = 1  # 每次梯度下降前，在环境中走几步收集新数据
    replay_buffer_max_length = 10000  # 经验回放池的最大容量，满了会覆盖最旧的数据
    batch_size = 64  # 每次从经验池抓取多少条数据喂给神经网络
    learning_rate = 1e-3  # Adam 优化器的学习率
    log_interval = 200  # 每隔 200 步在控制台打印一次 Loss 损失值
    eval_interval = 1000  # 每隔 1000 步暂停训练，运行几局测试来评估当前模型的真实水平

    # 实例化写的环境包装器，并通过TFPyEnvironment在底层将python环境吐出的所有数据转换为TF原生的tf.Tensor张量，并自动加上一个batch维度表示第几个batch
    py_env=TFAgentSimulationEnv()
    tf_env=tf_py_environment.TFPyEnvironment(py_env)

    # 实例化评估专用环境
    eval_py_env=TFAgentSimulationEnv()
    eval_tf_env=tf_py_environment.TFPyEnvironment(eval_py_env)

    # 构建Q神经网络（6x12x3）输入层6 6维观测向量.隐藏层12，输出层3 3维动作向量
    q_net=q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(12, )
    )

    # 优化器使用经典的Adam优化器，创建一个tf变量来记录当前是第几次迭代
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter=tf.Variable(0)

    #eplison参数在DQN中代表随机乱走的的机率，1代表闭着眼睛乱走，0代表完全听从神经网络的指挥来走。
    # 为了解决冷启动的问题，我们使用PolynomialDecay（本来是用来对学习率进行衰减用的）工具来对epsilon进行衰减
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
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda:epsilon_fn(train_step_counter)
    )
    agent.initialize()

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
    # 冷启动预热
    random_policy=random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),tf_env.action_spec())
    for _ in range(initial_collect_steps):
        collect_step(tf_env, random_policy, replay_buffer)

    # 构建高性能数据管道
    dateset=replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dateset)

    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    for _ in range(num_iterations):
        collect_step(tf_env, agent.collect_policy, replay_buffer)
        experience, unsend_info = next(iterator)
        loss_info = agent.train(experience)
        train_loss= loss_info.loss.numpy()

        step= agent.train_step_counter.numpy()
        if step%log_interval==0:
            print(f"步数(step): {step}|损失(loss): {train_loss:.4f}")


        if step%eval_interval==0:
            avg_return =compute_avg_return(eval_tf_env,agent.policy, num_episodes=5)
            print(f"--- 评估 --- 步数: {step} | 平均回报 (Avg Return): {avg_return:.2f}")

    # 在 20000 步训练结束后保存最终策略
    from tf_agents.policies import policy_saver
    import os

    print("正在保存控制组 A (纯 DQN) 的最优策略...")
    save_dir = os.path.join(os.getcwd(), 'baseline_dqn_policy')
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save(save_dir)
    print(f"模型已成功保存至: {save_dir}")

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

if __name__ == "__main__":
    train_tf_agents_dqn()






