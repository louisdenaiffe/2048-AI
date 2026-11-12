import tkinter as tk
import argparse

from game import Game2048, Game2048GUI
from environment import encode_board
from masked_dqn import MaskedDQN


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="dqn_2048")
args = parser.parse_args()

model = MaskedDQN.load(args.model)

root = tk.Tk()
game = Game2048(seed=42)
gui = Game2048GUI(root, game)


def choose_action(board):
    action, _ = model.predict(encode_board(board), deterministic=True) # masked: only valid moves
    return int(action)


gui.run_agent(choose_action, delay=150)
root.mainloop()
