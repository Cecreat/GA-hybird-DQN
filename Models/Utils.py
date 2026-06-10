from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


class Utils_stats:
    def compute_early_collision_rate(self,py_env, first_n_episodes=500):
        records = py_env.episode_records[:first_n_episodes]
        if len(records) == 0:
            return 0.0
        collision_count = sum(record["collision"] for record in records)
        return collision_count / len(records)
    def compute_success_rate(self,py_env):
        records = py_env.episode_records
        if len(records) == 0:
            return 0.0
        success_count = sum(record["success"] for record in records)
        return success_count / len(records)

    def compute_path_efficiency(self,py_env, first_n_episodes=None, last_n_episodes=None):
        """
        计算成功 episode 的路径效率。 Path Efficiency Ratio = actual_path_length / optimal_distance只统计 success episode 数值越接近 1 越好。
        """
        records = py_env.episode_records
        if first_n_episodes is not None:
            records = records[:first_n_episodes]

        if last_n_episodes is not None:
            records = records[-last_n_episodes:]

        ratios = []

        for record in records:
            if record.get("success", 0) != 1:
                continue

            ratio = record.get("path_efficiency_ratio", None)

            if ratio is None:
                continue

            ratios.append(float(ratio))

        if len(ratios) == 0:
            return {
                "success_count": 0,
                "mean_path_efficiency": None,
                "median_path_efficiency": None,
                "min_path_efficiency": None,
                "max_path_efficiency": None
            }

        ratios = np.array(ratios, dtype=np.float32)

        return {
            "success_count": int(len(ratios)),
            "mean_path_efficiency": float(np.mean(ratios)),
            "median_path_efficiency": float(np.median(ratios)),
            "min_path_efficiency": float(np.min(ratios)),
            "max_path_efficiency": float(np.max(ratios))
        }

    def compute_collision_rate_curve(self,py_env, first_n_episodes=500, window_size=50):
        """
        计算 Collision Rate Decay Curve。

        参数:
            py_env: TFAgentSimulationEnv
            first_n_episodes: 只统计前 N 个 episode，例如 500
            window_size: 每多少个 episode 统计一次碰撞率，例如 50

        返回:
            x: episode 横坐标
            y: collision rate 纵坐标
        """

        records = py_env.episode_records[:first_n_episodes]

        x = []
        y = []

        if len(records) == 0:
            return x, y

        for start in range(0, len(records), window_size):
            window = records[start:start + window_size]

            if len(window) == 0:
                continue

            collision_count = sum(record["collision"] for record in window)
            collision_rate = collision_count / len(window)

            # 横坐标用当前窗口的结束 episode 编号
            x.append(start + len(window))
            y.append(collision_rate)

        return x, y

    def plot_convergence_curve(
            self,
            plot_steps,
            plot_returns,
            plot_path,
            label="Baseline DQN",
            moving_average_window=5
    ):
        if len(plot_steps) == 0:
            return

        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))

        # 原始曲线
        plt.plot(
            plot_steps,
            plot_returns,
            label=f"{label} Raw",
            linewidth=1,
            alpha=0.4
        )

        # 滑动平均曲线
        if len(plot_returns) >= moving_average_window:
            smoothed_returns = np.convolve(
                plot_returns,
                np.ones(moving_average_window) / moving_average_window,
                mode="valid"
            )

            smoothed_steps = plot_steps[moving_average_window - 1:]

            plt.plot(
                smoothed_steps,
                smoothed_returns,
                label=f"{label} Moving Avg",
                linewidth=2
            )

        plt.xlabel("Training Steps")
        plt.ylabel("Average Return")
        plt.title("Convergence Speed: Evaluation Return over Training Steps")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def plot_collision_rate_curve(self,py_env,save_path,first_n_episodes=500,window_size=50,title="Early Collision Rate Decay Curve"):
        """
        绘制并保存 Collision Rate Decay Curve。
        """
        x, y = self.compute_collision_rate_curve(
            py_env,
            first_n_episodes=first_n_episodes,
            window_size=window_size
        )

        if len(x) == 0:
            print("没有 episode records，无法绘制 Collision Rate Decay Curve。")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker="o", label="Collision Rate")

        plt.xlabel("Episode")
        plt.ylabel("Collision Rate")
        plt.title(title)

        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Collision Rate Decay Curve 已保存至: {save_path}")



    def compute_avg_return(self,environment, policy, num_episodes=50, max_steps_per_episode=1000):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            step_count = 0
            while not time_step.is_last() and step_count < max_steps_per_episode:
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                step_count += 1
            total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def save_episode_records(self, py_env, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "episode",
            "collision",
            "success",
            "termination_reason",
            "steps",
            "episode_return",
            "actual_path_length",
            "optimal_distance",
            "path_efficiency_ratio"
        ]

        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in py_env.episode_records:
                row = {field: record.get(field, None) for field in fieldnames}
                writer.writerow(row)

        print(f"Episode records 已保存至: {save_path}")

