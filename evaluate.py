import numpy as np
from stable_baselines3 import DQN
from environment import Game2048Env


# This is for benchmarking models and seeing how they compare to each other.


def play_games(model, number_of_games=100):
    env = Game2048Env()  # Creates a Game2048Env instance 
    scores = []
    largest_tiles = []
    for game_number in range(number_of_games):  # we iterate through each game
        observation, info = env.reset(seed=game_number)  # starts a new game, and returns the game board + info about game stats, like the score, the largest tile...
        finished = False
        while not finished:
            action, _ = model.predict(observation, deterministic=True) 
            # Given this board (observation) what move should I choose?
            # action is 0 (up), 1(down), 2(left) or 3(right)
            observation, reward, terminated, truncated, info = env.step(action)  # this runs one timestep of the env's dynamics using the agent's chosen action.
            # Returns the board AFTER the move, points earned from merging (0 or -2), terminated (no moves remain), truncated (move limit reached), and updated stats
            finished = terminated or truncated # checks if the env is finished, if so ends the loop.
        scores.append(info["score"])
        largest_tiles.append(info["largest_tile"])
    env.close()
    return scores, largest_tiles


model = DQN.load("dqn_2048")
scores, largest_tiles = play_games(model)

print("Games:", len(scores))
print("Mean score:", np.mean(scores))
print("Median score:", np.median(scores))
print("Best score:", np.max(scores))
print("Mean largest tile:", np.mean(largest_tiles))
print("Best tile:", np.max(largest_tiles))

for tile in [64, 128, 256, 512, 1024, 2048]:
    percentage = np.mean(np.array(largest_tiles) >= tile) * 100
    print(f"Reached {tile}: {percentage:.1f}%")