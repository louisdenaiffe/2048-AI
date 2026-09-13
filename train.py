import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment import Game2048Env


def main():
    parser = argparse.ArgumentParser()  # Creates an instance of the ArgumentParser class and assigns it to the variable parser.
    parser.add_argument("--steps", type=int, default=200_000) # Adds a command-line argument --steps to the parser, which specifies the number of training steps for the model. The default value is set to 200,000.
    parser.add_argument("--model", default="dqn_2048") # Adds a command-line argument --model to the parser, which specifies the name of the model file to save or load. The default value is set to "dqn_2048".
    parser.add_argument("--continue-training", action="store_true", help="Continue training an existing model") # Adds a command-line argument --continue-training to the parser, which is a boolean flag. If this flag is provided, it indicates that the user wants to continue training an existing model instead of creating a new one.
    args = parser.parse_args() # Parses the command-line arguments and assigns the parsed values to the variable args. This allows access to the values of the --steps, --model, and --continue-training arguments.

    env = Monitor(Game2048Env()) # Creates an instance of the Game2048Env environment and wraps it with the Monitor class. The Monitor class is used to record training statistics, such as episode rewards and lengths, during the training process.
    model_path = Path(f"{args.model}.zip") # Model path

    try:
        if args.continue_training: # This will run if the --continue_training flag is provided when executing the script. 
            if not model_path.exists(): # if the model path doesn't exist, raises an error
                raise FileNotFoundError(f"Model not found: {model_path}")

            print(f"Continuing {model_path} for {args.steps:,} steps")

            model = DQN.load(
                args.model,
                env=env,
                device="auto",
            )
            reset_timesteps = False

        else: # this will run if the --continue_training flag is not provided when executing the script.
            print(f"Creating a new model for {args.steps:,} steps")

            model = DQN(
                policy="MlpPolicy", # The board has 16 numbers, so the input layer of the MLP policy will have 16 neurons. The output layer will have 4 neurons, corresponding to the 4 possible actions (up, down, left, right) in the game.
                env=env,
                learning_rate=0.0001, # Conservative learning rate, to avoid unstability in training.
                buffer_size=100_000, # Keeps many varied experiences in the replay buffer, without using excessive memory.
                learning_starts=10_000, # Collect 10k experiences before starting to train the mode, ensures that it doesnt start learning from a small, random number of experiences.
                batch_size=128, # Provides relatively stabke updates, while remaining computationally efficient.
                gamma=0.99, # Values long-term rewards strongly, because good early moves can enable much better tiles later.
                train_freq=4, # Trains once every four environment moves. This reduces computation and prevents excessive learning from very similar consecutive experiences.
                gradient_steps=1, # Performs one neural-network update whenever training occurs—a stable, efficient baseline.
                exploration_fraction=0.3, # Gradually transitions from random exploration to learned behavior during the first 30% of training.
                exploration_initial_eps=1.0, # Starts completely random so the replay buffer receives varied experiences.
                exploration_final_eps=0.05, # Keeps 5% randomness, allowing occasional discovery of new situations instead of permanently following one strategy.
                verbose=1,
                device="auto", # Automatically uses GPU if available, otherwise defaults to CPU
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