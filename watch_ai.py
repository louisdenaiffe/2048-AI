import tkinter as tk
import numpy as np
from stable_baselines3 import DQN

from game import Game2048, Game2048GUI


model = DQN.load("dqn_2048")

root = tk.Tk()
game = Game2048(seed=42)
gui = Game2048GUI(root, game)


def choose_action(board):
    observation = np.zeros_like(board, dtype=np.float32)
    nonzero = board > 0
    observation[nonzero] = np.log2(board[nonzero])
    observation = observation.flatten()

    action, _ = model.predict(observation, deterministic=True)
    action = int(action)

    # Your GUI rejects blocked moves, while the training environment permits
    # them with a -2 reward. Choose another valid move only when necessary.
    valid_actions = game.get_valid_actions()

    if action not in valid_actions:
        action = int(game.rng.choice(valid_actions))

    return action


gui.run_agent(choose_action, delay=150)
root.mainloop()