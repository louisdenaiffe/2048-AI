import csv
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from evaluate import play_games


# A callback is an object SB3 calls at certain moments of model.learn(): when training starts, after every step, when it ends.
# This one regularly pauses training to play some evaluation games, so we can follow how the REAL game score evolves
# (SB3's own logs only show the reward, which is log2-scaled and so not comparable with evaluate.py or the baselines).


TILES_TO_TRACK = [256, 512, 1024, 2048]


class ScoreEvalCallback(BaseCallback): # BaseCallback is SB3's parent class for callbacks: we fill in the moments we care about.

    def __init__(self, eval_freq, n_games, csv_path, best_model_path, append=False, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq # evaluate every eval_freq training steps
        self.n_games = n_games # number of evaluation games each time (always seeds 0 to n_games-1, so always the same games)
        self.csv_path = csv_path # where the results are saved, one row per evaluation
        self.best_model_path = best_model_path # where the best model so far is saved
        self.append = append # True when continuing a training run: add to the existing CSV instead of starting a new one
        self.best_mean_score = -np.inf # best mean score so far (-infinity, so the first evaluation is always "the best")
        self.last_eval_step = None # step of the last evaluation, to avoid evaluating twice at the same step

    def _on_training_start(self):
        # Called once, at the start of model.learn().
        if self.append and self.csv_path.exists():
            # Continuing a run: keep the existing results, and remember the best score so far,
            # so _best.zip isn't replaced by a worse model.
            with open(self.csv_path, newline="") as file:
                previous_scores = [float(row["mean_score"]) for row in csv.DictReader(file)]
            if previous_scores:
                self.best_mean_score = max(previous_scores)
        else:
            # New run: create the CSV file with only its header row (overwrites the results of an older run with the same name).
            with open(self.csv_path, "w", newline="") as file:
                csv.writer(file).writerow(["timesteps", "mean_score", "median_score", "best_tile"] + [f"reached_{tile}" for tile in TILES_TO_TRACK])
            self.evaluate() # evaluates the untrained model too, so the curve starts at step 0

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
        start = time.time()
        scores, largest_tiles = play_games(self.model, number_of_games=self.n_games)
        # self.model is the model being trained (SB3 sets it on the callback). play_games() uses deterministic, masked moves.
        largest_tiles = np.array(largest_tiles)
        mean_score = float(np.mean(scores))
        reached = {tile: float(np.mean(largest_tiles >= tile) * 100) for tile in TILES_TO_TRACK}
        # e.g. {256: 45.0, 512: 10.0, 1024: 0.0, 2048: 0.0}: percentage of games that reached each tile

        # Adds a row to the CSV file ("a" = append mode: writes at the end, without erasing what's there).
        with open(self.csv_path, "a", newline="") as file:
            csv.writer(file).writerow(
                [self.num_timesteps, mean_score, float(np.median(scores)), int(largest_tiles.max())] + [reached[tile] for tile in TILES_TO_TRACK]
            )

        # Also adds the numbers to SB3's log table, printed during training (and to TensorBoard, if it is ever used).
        self.logger.record("eval/mean_score", mean_score)
        for tile in TILES_TO_TRACK[:3]: # 256, 512, 1024 (2048 would only add a column of zeros for now)
            self.logger.record(f"eval/reached_{tile}_%", reached[tile])

        new_best = mean_score > self.best_mean_score
        if new_best: # keeps a copy of the best model seen so far (the last model isn't always the best one)
            self.best_mean_score = mean_score
            self.model.save(self.best_model_path)

        if self.verbose:
            print(
                f"[eval] step {self.num_timesteps:,}: mean score {mean_score:.0f}, "
                + ", ".join(f"{tile}: {reached[tile]:.0f}%" for tile in TILES_TO_TRACK[:3])
                + (" (new best, saved)" if new_best else "")
                + f" [{time.time() - start:.1f}s]"
            )
        self.last_eval_step = self.num_timesteps
