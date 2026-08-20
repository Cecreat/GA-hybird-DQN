import numpy as np
from pathlib import Path
from collections import deque

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

BASE_DIR = Path(__file__).resolve().parent

try:
    from Env import SimulationEnv, NUM_RAYS
except ModuleNotFoundError:
    from Env import SimulationEnv, NUM_RAYS


SINGLE_FRAME_OBS_DIM = NUM_RAYS + 2
DEFAULT_FRAME_STACK = 4


class TFAgentSimulationEnv(py_environment.PyEnvironment):
    """
    将 Python 原生 SimulationEnv 包装为 TF-Agents 可用的 PyEnvironment。

    关键设计：
        - SimulationEnv 仍然只输出单帧 13 维状态；
        - 本包装器负责维护最近 frame_stack 帧；
        - TF-Agents / DQN 看到的是 frame_stack * 13 维 observation。

    默认 frame_stack=4，因此 observation shape = (52,)。

    seed:
        - None: 训练时默认使用非固定随机序列；
        - int: fixed-seed evaluation 时使用固定环境序列。
    """

    def __init__(self, seed=None, frame_stack=DEFAULT_FRAME_STACK):
        self.seed_value = seed
        self.frame_stack = int(frame_stack)
        if self.frame_stack <= 0:
            raise ValueError(f"frame_stack 必须为正整数，当前为: {frame_stack}")

        self.single_frame_obs_dim = SINGLE_FRAME_OBS_DIM
        self.stacked_obs_dim = self.single_frame_obs_dim * self.frame_stack

        self.last_info = {}
        self.episode_count = 0
        self.collision_count = 0
        self.success_count = 0
        self.episode_records = []
        self.current_episode_steps = 0
        self.current_episode_return = 0.0

        self._frame_buffer = deque(maxlen=self.frame_stack)

        super().__init__()

        self._env = SimulationEnv(seed=seed)

        self.action_mapping = {
            0: -1.0,
            1: -0.5,
            2: 0.0,
            3: 0.5,
            4: 1.0,
        }

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=4,
            name="action"
        )

        single_min = np.array([0.0] * NUM_RAYS + [-1.0, 0.0], dtype=np.float32)
        single_max = np.array([1.0] * NUM_RAYS + [1.0, 1.0], dtype=np.float32)

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.stacked_obs_dim,),
            dtype=np.float32,
            minimum=np.tile(single_min, self.frame_stack).astype(np.float32),
            maximum=np.tile(single_max, self.frame_stack).astype(np.float32),
            name=f"observation_framestack{self.frame_stack}"
        )

        self._state = np.zeros((self.stacked_obs_dim,), dtype=np.float32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def seed(self, seed=None):
        return self.set_seed(seed)

    def set_seed(self, seed=None):
        self.seed_value = seed
        if hasattr(self, "_env") and self._env is not None:
            self._env.set_seed(seed)
        return [seed]

    def reset_episode_records(self):
        self.episode_records.clear()
        self.episode_count = 0
        self.collision_count = 0
        self.success_count = 0
        self.current_episode_steps = 0
        self.current_episode_return = 0.0
        self.last_info = {}

    def _to_single_frame_state(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.single_frame_obs_dim:
            raise ValueError(
                f"SimulationEnv 应输出 {self.single_frame_obs_dim} 维单帧状态，"
                f"但得到 {state.shape[0]} 维。"
            )
        return state

    def _init_frame_stack(self, single_state):
        self._frame_buffer.clear()
        single_state = self._to_single_frame_state(single_state)
        for _ in range(self.frame_stack):
            self._frame_buffer.append(single_state.copy())
        self._state = self._get_stacked_state()

    def _append_frame(self, single_state):
        self._frame_buffer.append(self._to_single_frame_state(single_state).copy())
        self._state = self._get_stacked_state()

    def _get_stacked_state(self):
        if len(self._frame_buffer) != self.frame_stack:
            raise RuntimeError(
                f"frame buffer 长度错误: {len(self._frame_buffer)}，"
                f"期望 {self.frame_stack}。"
            )
        return np.concatenate(list(self._frame_buffer), axis=0).astype(np.float32)

    def get_latest_single_frame(self):
        """
        返回当前 stacked observation 中最新的一帧 13 维状态。
        主要用于调试或 GA heuristic controller。
        """
        return self._state[-self.single_frame_obs_dim:].copy()

    def _reset(self):
        single_state = self._env.reset()
        self._init_frame_stack(single_state)
        self._episode_ended = False
        self.current_episode_steps = 0
        self.current_episode_return = 0.0
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        action = int(np.asarray(action).item())
        real_action = self.action_mapping[action]

        next_single_state, reward, done, info = self._env.step(real_action)
        self._append_frame(next_single_state)
        self.last_info = info

        reward = np.float32(reward)
        self.current_episode_steps += 1
        self.current_episode_return += float(reward)

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
                "seed": self.seed_value,
                "episode": self.episode_count,
                "collision": int(is_collision),
                "success": int(is_success),
                "termination_reason": info.get("termination_reason", "unknown"),
                "steps": int(self.current_episode_steps),
                "episode_return": float(self.current_episode_return),
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
