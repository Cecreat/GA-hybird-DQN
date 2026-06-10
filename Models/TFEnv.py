import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
# TF-Agents 相关组件
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
BASE_DIR = Path(__file__).resolve().parent
from Env import SimulationEnv

"""
封装Python环境为TF环境
定义_action_spec 和  _observation_spec、以及它们的获取接口
编写环境_reset函数
编写_step函数
返回next_state, reward and discount

"""
class TFAgentSimulationEnv(py_environment.PyEnvironment):
    def __init__(self):
        self.last_info = {}
        self.episode_count = 0
        self.collision_count = 0
        self.success_count = 0
        self.episode_records = []
        # 调用父类构造函数初始化基础环境所需的变量
        super().__init__()
        # 实例化写好的python环境
        self._env = SimulationEnv()
        # TF在处理离散动作的时候默认从0开始正整数索引，所以需要通过映射解决“语言不同”的问题
        self.action_mapping = {
            0: -1.0,
            1: -0.5,
            2: 0.0,
            3: 0.5,
            4: 1.0,
        }
        # 定义动作空间的规格：  shape=()表明是标量，dtype=np.int32表明必须是32位整数，minimum=0,maximum=4明确范围 本质是神经网络的输出空间
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=4,
            name='action'
        )
        # 定义观察空间的规格  shape=(13,)表明是一个包含十三个元素的一维数组 本质是神经网络的输入空间
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(13,),
            dtype=np.float32,
            minimum=np.array([0.0] * 11 + [-1.0, 0.0], dtype=np.float32),#[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
            maximum=np.array([1.0] * 11 + [1.0, 1.0], dtype=np.float32), #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            name='observation'
        )

        self._state = np.zeros((13,), dtype=np.float32)
        self._episode_ended = False
    # 获取两个规格的接口
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    # 触发底层真实的物理环境进行重置
    def _reset(self):
        # 调用python环境中的reset重置环境并获取新的状态（12维向量，11个射线的感知结果和1个与目标点的角度差）
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
        action = int(np.asarray(action).item())
        real_action = self.action_mapping[action]
        result = self._env.step(real_action)
        next_state, reward, done, info = result
        self._state = np.array(next_state, dtype=np.float32)
        self.last_info = info

        reward = np.float32(reward)

        if done:
            self._episode_ended = True

            is_collision = bool(info.get("collision", False))
            is_success = bool(info.get("success", False))

            self.episode_count += 1

            if is_collision:
                self.collision_count += 1

            if is_success:
                self.success_count += 1

            self.episode_records.append({
                "episode": self.episode_count,
                "collision": int(is_collision),
                "success": int(is_success),
                "termination_reason": info.get("termination_reason", "unknown"),
                "actual_path_length": info.get("actual_path_length", None),
                "optimal_distance": info.get("optimal_distance", None),
                "path_efficiency_ratio": info.get("path_efficiency_ratio", None)
            })

            return ts.termination(self._state, reward=reward)

        return ts.transition(
            self._state,
            reward=reward,
            discount=np.float32(0.99)
        )