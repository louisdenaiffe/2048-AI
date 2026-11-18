import tkinter as tk
import argparse

from game import Game2048, Game2048GUI
from environment import encode_board
from masked_dqn import MaskedDQN
from model_files import find_model_to_load


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="dqn_2048", help="A model name (loads models/<name>/model.zip) or the path of a model file")
parser.add_argument("--lookahead", type=int, nargs="?", const=1, default=0, choices=[0, 1, 2],
                    help="Choose moves with a lookahead search: --lookahead = 1 move, --lookahead 2 = 2 moves (see lookahead.py)")
args = parser.parse_args()

model = MaskedDQN.load(find_model_to_load(args.model))
model.lookahead_depth = args.lookahead # 0 by default: greedy play

root = tk.Tk()
game = Game2048(seed=42)
gui = Game2048GUI(root, game)


def choose_action(board):
    action, _ = model.predict(encode_board(board), deterministic=True) # masked: only valid moves
    return int(action)


gui.run_agent(choose_action, delay=150)
root.mainloop()
