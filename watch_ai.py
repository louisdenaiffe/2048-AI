import tkinter as tk
from stable_baselines3 import DQN

from game import Game2048, Game2048GUI
from environment import encode_board


model = DQN.load("dqn_2048")

root = tk.Tk()
game = Game2048(seed=42)
gui = Game2048GUI(root, game)


def choose_action(board):
    action, _ = model.predict(encode_board(board), deterministic=True)
    return int(action)


gui.run_agent(choose_action, delay=150)
root.mainloop()
