import os
import tensorflow as tf
import pygame


from Models.DQN_agent import TFAgentSimulationEnv
from tf_agents.environments import tf_py_environment


def play_saved_policy(policy_dir, num_episodes=5):


    print(f"正在加载预训练模型: {policy_dir} ...")
    if not os.path.exists(policy_dir):
        print("错误：未找到模型文件夹！")
        return

    saved_policy = tf.saved_model.load(policy_dir)
    print("✅ 模型加载成功！准备下发至物理环境...\n")


    py_env = TFAgentSimulationEnv()
    env = tf_py_environment.TFPyEnvironment(py_env)

    total_steps = 0
    total_reward = 0.0
    for episode in range(num_episodes):

        time_step = env.reset()
        episode_reward = 0.0
        survival_steps = 0

        print(f"🎬 --- 开始第 {episode + 1}/{num_episodes} 局演示 ---")


        while not time_step.is_last():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            policy_step = saved_policy.action(time_step)


            time_step = env.step(policy_step.action)


            episode_reward += time_step.reward.numpy()[0]
            survival_steps += 1


            py_env._env.render()
        total_steps +=survival_steps
        total_reward += episode_reward
        print(f"🏁 第 {episode + 1} 局结束 | 存活步数: {survival_steps} | 总得分: {episode_reward:.2f}")
    avg_step=total_steps/num_episodes
    avg_reward=total_reward/num_episodes
    print(f"Avg steps: {avg_step} | Avg reward: {avg_reward:.2f}")
    print("\n演示完毕，正在关闭环境...")
    pygame.quit()


if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), '../hybrid_ga_rl_policy')
    play_saved_policy(model_path, num_episodes=20)