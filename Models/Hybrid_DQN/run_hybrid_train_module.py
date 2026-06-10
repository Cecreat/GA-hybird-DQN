import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from Hybrid_train_module import Hybrid_Train


def main():
    trainer = Hybrid_Train()
    trainer.train_hybrid_agent()


if __name__ == "__main__":
    main()