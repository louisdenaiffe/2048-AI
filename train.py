import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment import Game2048Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--model", default="dqn_2048")
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Continue training an existing model",
    )
    args = parser.parse_args()

    env = Monitor(Game2048Env())
    model_path = Path(f"{args.model}.zip")

    try:
        if args.continue_training:
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            print(f"Continuing {model_path} for {args.steps:,} steps")

            model = DQN.load(
                args.model,
                env=env,
                device="auto",
            )
            reset_timesteps = False

        else:
            print(f"Creating a new model for {args.steps:,} steps")

            model = DQN(
                policy="MlpPolicy",
                env=env,
                learning_rate=0.0001,
                buffer_size=100_000,
                learning_starts=10_000,
                batch_size=128,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=1,
                device="auto",
            )
            reset_timesteps = True

        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
        )

        model.save(args.model)
        print(f"Saved model to {model_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()