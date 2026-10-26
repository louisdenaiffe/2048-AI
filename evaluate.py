import numpy as np
from stable_baselines3 import DQN
from environment import Game2048Env


# This is for evaluating models and seeing how they compare to each other.


def play_games(model, number_of_games=100):
    env = Game2048Env()  # Creates a Game2048Env instance 
    scores = []
    largest_tiles = []
    games_truncated = 0
    games_terminated = 0
    blocked_moves = 0
    total_blocked_moves = []
    for game_number in range(number_of_games):  # we iterate through each game
        observation, info = env.reset(seed=game_number)  # starts a new game, and returns the game board + info about game stats, like the score, the largest tile...
        finished = False
        while not finished:
            action, _ = model.predict(observation, deterministic=True) 
            # Given this board (observation) what move should I choose?
            # action is 0 (up), 1(down), 2(left) or 3(right)
            observation, reward, terminated, truncated, blocked_move, info = env.step(action)  # this runs one timestep of the env's dynamics using the agent's chosen action.
            # Returns the board AFTER the move, points earned from merging (0 or -2), terminated (no moves remain), truncated (move limit reached), and updated stats
            # These are actually all defined in environment.py, so check that out.
            if blocked_move: blocked_moves += 1
            finished = terminated or truncated # checks if the env is finished, if so ends the loop.
        scores.append(info["score"]) # appends the score for that game (also called episode) to a score list
        largest_tiles.append(info["largest_tile"]) # same with the largest tile
        # So now we have the score and the largest tile from each game
        games_truncated += 1 if truncated else 0
        games_terminated += 1 if terminated else 0
        total_blocked_moves.append(blocked_moves)
    env.close()
    return scores, largest_tiles, games_terminated, games_truncated, total_blocked_moves


model = DQN.load("dqn_2048")
scores, largest_tiles, games_terminated, games_truncated, total_blocked_moves= play_games(model) 
# calls the function and passes on the scores and largest_tiles lists

# Benchmarking: mean, median, max, mean of the largest tiles, best tile.

print("Games:", len(scores))
print("Mean score:", np.mean(scores))
print("Median score:", np.median(scores))
print("Best score:", np.max(scores))
print("Mean largest tile:", np.mean(largest_tiles))
print("Best tile:", np.max(largest_tiles))
print("Percentage of games with a natural game over:", (games_terminated / len(scores)) * 100, "%")
print("Percentage of games finished because the max amount of moves was reached:", (games_truncated / len(scores)) * 100, "%")
print("Mean blocked moves per game:", np.mean(total_blocked_moves))

for tile in [64, 128, 256, 512, 1024, 2048]:
    percentage = np.mean(np.array(largest_tiles) >= tile) * 100
    print(f"Reached {tile}: {percentage:.1f}%")