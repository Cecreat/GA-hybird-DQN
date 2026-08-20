import os
from pathlib import Path

# 必须放在 tensorflow / tf_agents 被 import 之前
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from Evaluation_module import DQNEvaluator


BASE_DIR = Path(__file__).resolve().parent


def main():
    # Pure DQN run directory
    run_dir = BASE_DIR / "runs" / "dqn_obs13_fc64_64_act5_15obstacles"

    # 使用 best checkpoint，而不是最后一个 latest checkpoint
    checkpoint_dir = run_dir / "best_checkpoint"

    # fixed-seed 评估输出目录
    fixed_seed_eval_dir = run_dir / "fixed_seed_eval"

    evaluator = DQNEvaluator(
        checkpoint_dir=checkpoint_dir,
        fc_layer_params=(64, 64),
        learning_rate=5e-4
    )

    result = evaluator.evaluate_policy_fixed_seeds(
        seed_list=[0, 1, 2, 3, 4],
        episodes_per_seed=500,
        max_steps_per_episode=1000,
        save_dir=fixed_seed_eval_dir
    )

    evaluator.print_fixed_seed_summary(
        result,
        prefix="Fixed-Seed DQN Policy Evaluation"
    )

    print("\n固定随机种子评估完成。")
    print(f"Fixed-seed evaluation results saved to: {fixed_seed_eval_dir}")
    print(f"Per-seed summary: {fixed_seed_eval_dir / 'FixedSeed_Evaluation_PerSeed_Summary.csv'}")
    print(f"Mean/std summary: {fixed_seed_eval_dir / 'FixedSeed_Evaluation_MeanStd.csv'}")


if __name__ == "__main__":
    main()
