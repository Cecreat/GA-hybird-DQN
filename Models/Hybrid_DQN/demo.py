import os
import time
import pygame
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common




from Hybrid_train_module import TFAgentSimulationEnv


CHECKPOINT_DIR = r"D:\MSc_Project\Models\Hybrid_DQN\runs\hybrid_ga_heuristic_dqn_obs13_fc64_64_act5_5obs(2)\checkpoint"  # 改成你的 checkpoint 文件夹
NUM_EPISODES = 50
FPS = 30

FC_LAYER_PARAMS = (64, 64)      # 必须和训练时一致
LEARNING_RATE = 5e-4            # 恢复模型时影响不大，但建议和训练一致



py_env = TFAgentSimulationEnv()

tf_env = tf_py_environment.TFPyEnvironment(py_env)


q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=FC_LAYER_PARAMS
)


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()



latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)

if latest_ckpt is None:
    raise FileNotFoundError(f"没有在该路径找到 checkpoint: {CHECKPOINT_DIR}")

print(f"Restoring checkpoint from: {latest_ckpt}")

checkpointer = common.Checkpointer(
    ckpt_dir=CHECKPOINT_DIR,
    agent=agent,
    policy=agent.policy,
    global_step=train_step_counter
)

checkpointer.initialize_or_restore()

print("Checkpoint restored successfully.")


def try_render(env):
    """
    安全渲染：
    1. 如果当前环境自己实现了 render()，就调用。
    2. 如果只是继承了 TF-Agents 的默认 render()，会抛 NotImplementedError，这里直接跳过。
    3. 如果环境内部包了一层真正的 pygame 环境，则尝试调用内部环境的 render()。
    """

    # 先尝试当前 env.render()
    try:
        env.render()
        return
    except NotImplementedError:
        pass
    except AttributeError:
        pass

    # 再尝试常见的内部环境变量名
    possible_inner_env_names = [
        "_env",
        "env",
        "py_env",
        "pygame_env",
        "simulation_env",
        "base_env",
        "sim_env"
    ]

    for name in possible_inner_env_names:
        inner_env = getattr(env, name, None)

        if inner_env is not None and hasattr(inner_env, "render"):
            try:
                inner_env.render()
                return
            except NotImplementedError:
                pass
            except AttributeError:
                pass


def run_demo():
    pygame.init()
    clock = pygame.time.Clock()

    returns = []
    steps_list = []

    for episode in range(1, NUM_EPISODES + 1):
        time_step = tf_env.reset()

        episode_return = 0.0
        episode_steps = 0

        while not time_step.is_last().numpy()[0]:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return


            action_step = agent.policy.action(time_step)
            action = action_step.action

            time_step = tf_env.step(action)

            reward = float(time_step.reward.numpy()[0])
            episode_return += reward
            episode_steps += 1

            try_render(py_env)

            clock.tick(FPS)

        returns.append(episode_return)
        steps_list.append(episode_steps)

        print(
            f"Episode {episode:02d} | "
            f"Return: {episode_return:.2f} | "
            f"Steps: {episode_steps}"
        )

        time.sleep(0.5)

    pygame.quit()



if __name__ == "__main__":
    run_demo()