import tkinter as tk
from stable_baselines3 import DQN
import argparse

from game import Game2048, Game2048GUI
from environment import encode_board, choose_masked_action


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="dqn_2048")
args = parser.parse_args()

model = DQN.load(args.model)

root = tk.Tk()
game = Game2048(seed=42)
gui = Game2048GUI(root, game)


def choose_action(board):
    return choose_masked_action(model, encode_board(board), game.get_valid_actions())


gui.run_agent(choose_action, delay=150)
root.mainloop()
