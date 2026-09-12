import numpy as np
from stable_baselines3 import DQN
from environment import Game2048Env


def play_games(model, number_of_games=100):
    env = Game2048Env()
    scores = []
    largest_tiles = []
    for game_number in range(number_of_games):
        observation, info = env.reset(seed=game_number)
        finished = False
        while not finished:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            finished = terminated or truncated
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