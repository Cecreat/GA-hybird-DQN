import os
from pathlib import Path

# 必须放在 tensorflow / tf_agents 被 import 之前
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from Hybrid_evaluation_module import DQNEvaluator


BASE_DIR = Path(__file__).resolve().parent


def main():
    run_dir = BASE_DIR / "runs" / "hybrid_ga_heuristic_dqn_obs13_fc64_64_act5_(2)"

    checkpoint_dir = run_dir / "checkpoint"
    final_eval_records_path = run_dir / "Final_Hybrid_Evaluation_Records_show.csv"

    evaluator = DQNEvaluator(
        checkpoint_dir=checkpoint_dir,
        fc_layer_params=(64, 64),
        learning_rate=5e-4
    )

    summary = evaluator.evaluate_policy(
        num_episodes=5,
        max_steps_per_episode=1000
    )

    evaluator.print_summary(
        summary,
        prefix="Final DQN Policy Evaluation"
    )

    evaluator.save_eval_records(final_eval_records_path)

    print("\n评估完成。")
    print(f"Final evaluation records saved to: {final_eval_records_path}")


if __name__ == "__main__":
    main()