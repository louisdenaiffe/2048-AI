import csv
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from evaluate import play_games


# A callback is an object SB3 calls at certain moments of model.learn(): when training starts, after every step, when it ends.
# This one regularly pauses training to play some evaluation games, so we can follow how the REAL game score evolves
# (SB3's own logs only show the reward, which is log2-scaled and so not comparable with evaluate.py or the baselines).
# It always records greedy play (eval.csv), and optionally also play with a 1-move lookahead (eval_lookahead.csv),
# since models trained with lookahead can play badly greedy but very well with lookahead.


TILES_TO_TRACK = [256, 512, 1024, 2048]
CSV_HEADER = ["timesteps", "mean_score", "median_score", "best_tile"] + [f"reached_{tile}" for tile in TILES_TO_TRACK]


class ScoreEvalCallback(BaseCallback): # BaseCallback is SB3's parent class for callbacks: we fill in the moments we care about.

    def __init__(self, eval_freq, n_games, csv_path, lookahead_csv_path=None, append=False, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq # evaluate every eval_freq training steps
        self.n_games = n_games # number of evaluation games each time (always seeds 0 to n_games-1, so always the same games)
        # One results file per way of playing: lookahead depth -> CSV path. Depth 0 (greedy) is always recorded,
        # depth 1 only if lookahead_csv_path is given (train.py --eval-lookahead).
        self.csv_paths = {0: csv_path}
        if lookahead_csv_path is not None:
            self.csv_paths[1] = lookahead_csv_path
        self.append = append # True when continuing a training run: add to the existing CSVs instead of starting new ones
        self.last_eval_step = None # step of the last evaluation, to avoid evaluating twice at the same step

    def _on_training_start(self):
        # Called once, at the start of model.learn().
        for depth, path in self.csv_paths.items():
            # Continuing a run (append, with an existing CSV): nothing to do, new rows are simply added at the end.
            if not (self.append and path.exists()):
                # New run (or a new kind of curve for a continued run, e.g. the first --eval-lookahead on lookahead_v1):
                # create the CSV file with only its header row (overwrites the results of an older run with the same name),
                # and evaluate the starting model, so this curve has a starting point.
                with open(path, "w", newline="") as file:
                    csv.writer(file).writerow(CSV_HEADER)
                self.evaluate_one(depth, path)
                self.last_eval_step = self.num_timesteps

    def _on_step(self):
        # Called after every training step. self.num_timesteps is the model's total number of steps so far
        # (it keeps counting when continuing a run, so the curve continues too).
        if self.num_timesteps % self.eval_freq == 0: # % is the remainder of the division: 0 every eval_freq steps
            self.evaluate()
        return True # returning False would stop the training

    def _on_training_end(self):
        # Called once, at the end of model.learn(): one last evaluation, unless we just did one at this exact step.
        if self.last_eval_step != self.num_timesteps:
            self.evaluate()

    def evaluate(self):
        for depth, csv_path in self.csv_paths.items():
            self.evaluate_one(depth, csv_path)
        self.last_eval_step = self.num_timesteps

    def evaluate_one(self, depth, csv_path):
        start = time.time()
        self.model.lookahead_depth = depth # 0 = greedy, 1 = 1-move lookahead (see predict() in masked_dqn.py)
        try:
            scores, largest_tiles = play_games(self.model, number_of_games=self.n_games)
            # self.model is the model being trained (SB3 sets it on the callback). play_games() uses deterministic, masked moves.
        finally:
            self.model.lookahead_depth = 0 # back to the default, whatever happens
        largest_tiles = np.array(largest_tiles)
        mean_score = float(np.mean(scores))
        reached = {tile: float(np.mean(largest_tiles >= tile) * 100) for tile in TILES_TO_TRACK}
        # e.g. {256: 45.0, 512: 10.0, 1024: 0.0, 2048: 0.0}: percentage of games that reached each tile

        # Adds a row to the CSV file ("a" = append mode: writes at the end, without erasing what's there).
        with open(csv_path, "a", newline="") as file:
            csv.writer(file).writerow(
                [self.num_timesteps, mean_score, float(np.median(scores)), int(largest_tiles.max())] + [reached[tile] for tile in TILES_TO_TRACK]
            )

        # Also adds the numbers to SB3's log table, printed during training (and to TensorBoard, if it is ever used).
        prefix = "eval" if depth == 0 else "eval_lookahead"
        self.logger.record(f"{prefix}/mean_score", mean_score)
        for tile in TILES_TO_TRACK:
            self.logger.record(f"{prefix}/reached_{tile}_%", reached[tile])

        if self.verbose:
            print(
                f"[{prefix}] step {self.num_timesteps:,}: mean score {mean_score:.0f}, "
                + ", ".join(f"{tile}: {reached[tile]:.0f}%" for tile in TILES_TO_TRACK)
                + f" [{time.time() - start:.1f}s]"
            )
