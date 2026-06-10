import os

# 必须放在 tensorflow / tf_agents 被 import 之前
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from TrainModule import Train


def main():
    trainer = Train()
    trainer.train_tf_agents_dqn()


if __name__ == "__main__":
    main()