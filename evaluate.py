import numpy as np
import argparse
from environment import Game2048Env
from masked_dqn import MaskedDQN
from model_files import find_model_to_load


# This is for evaluating models and seeing how they compare to each other.


def play_games(model, number_of_games=100):
    env = Game2048Env()  # Creates a Game2048Env instance 
    scores = []
    largest_tiles = []
    for game_number in range(number_of_games):  # we iterate through each game
        observation, info = env.reset(seed=game_number)  # starts a new game, and returns the game board + info about game stats, like the score, the largest tile...
        finished = False
        while not finished:
            action, _ = model.predict(observation, deterministic=True)
            # Given this board (observation) what move should I choose? Only valid moves are considered, so the agent can't get stuck.
            # action is 0 (up), 1(down), 2(left) or 3(right)
            observation, reward, terminated, truncated, info = env.step(action)  # this runs one timestep of the env's dynamics using the agent's chosen action.
            # Returns the board AFTER the move, points earned from merging (0 if nothing merged), terminated (no moves remain), truncated (move limit reached), and updated stats
            # These are actually all defined in environment.py, so check that out.
            finished = terminated or truncated # checks if the env is finished, if so ends the loop.
        scores.append(info["score"]) # appends the score for that game (also called episode) to a score list
        largest_tiles.append(info["largest_tile"]) # same with the largest tile
        # So now we have the score and the largest tile from each game
    env.close()
    return scores, largest_tiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dqn_2048", help="A model name (loads models/<name>/model.zip) or the path of a model file")
    args = parser.parse_args()

    model = MaskedDQN.load(find_model_to_load(args.model)) # MaskedDQN can also load models trained with plain DQN, and plays them with masking
    scores, largest_tiles = play_games(model)
    # calls the function and passes on the scores and largest_tiles lists

    # Benchmarking: mean, median, max, mean of the largest tiles, best tile.

    print("Games:", len(scores))
    print("Mean score:", np.mean(scores))
    print("Median score:", np.median(scores))
    print("Best score:", np.max(scores))
    print("Mean largest tile:", np.mean(largest_tiles))
    print("Best tile:", np.max(largest_tiles))

    for tile in [64, 128, 256, 512, 1024, 2048]:
        percentage = np.mean(np.array(largest_tiles) >= tile) * 100
        print(f"Reached {tile}: {percentage:.1f}%")


if __name__ == "__main__":
    # __name__ is "__main__" only when this file is run directly (python evaluate.py).
    # When another file imports it (from evaluate import play_games), __name__ is "evaluate", so main() doesn't run.
    main()