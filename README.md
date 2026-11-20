# 2048-AI

The goal of this project is to train an AI to play 2048 using Deep Q-Learning (DQN), a form of reinforcement learning.
2048 is a sliding-tile puzzle where you merge equal-numbered tiles to create the 2048 tile.

[![nfy8A8X.md.png](https://iili.io/nfy8A8X.md.png)](https://freeimage.host/i/nfy8A8X)

**Result:** the final agent reaches the 2048 tile in **98% of games it has never seen**, and the 4096 tile in about half of them (mean score ~51,000).
It combines a neural network trained with DQN and an expectimax search that uses that network to plan a few moves ahead.

<p align="center">
  <img src="figures/ai_playing.gif" alt="The agent playing a full game of 2048 and reaching the 4096 tile, with a final score of 80,352" width="370">
  <br>
  <em>The final agent (lookahead_v1, adaptive 2-to-4-move search) playing seed 0: it reaches 4096 with a score of 80,352.<br>
  Most of the game is sped up; the last 40 moves are shown one by one.</em>
</p>

# Table of contents
 - [Results](#results)
 - [How the model works](#how-the-model-works)
   - [The network](#the-network-learned-with-dqn)
   - [The search](#the-search-expectimax)
   - [Training](#training)
 - [Evaluation method](#evaluation-method)
 - [Evolution of the model](#evolution-of-the-model)
 - [Project structure](#project-structure)
 - [How to use](#how-to-use)
 - [Glossary](#glossary)


## Results

100 games each, on fresh seeds (games never used to make any decision during the project), model `lookahead_v1`:

| How the agent plays | Mean score | Reached 2048 | Reached 4096 | Time per game (CPU) |
|---|---|---|---|---|
| Network alone (greedy) | ~1,500 | 0% | 0% | instant |
| 2-move search | 30,424 | 82% | — | ~4 s |
| **Adaptive search (2 to 4 moves)** | **51,222** | **98%** | **54%** | ~3 min |

For comparison, the best model trained with plain DQN (no search, `empty_v2`) averages ~7,500 points and never reaches 2048.


## How the model works

The model has two parts: a **neural network** that learns to judge boards, and a **search** that uses those judgments to choose each move.

### The network (learned with DQN)

- **Input:** the board as 256 numbers. Each of the 16 cells gets 16 slots (empty, 2, 4, 8, ... 32768), with a 1 in the slot of its tile and 0 everywhere else (one-hot encoding).
- **Hidden layers:** 2 layers of 256 neurons, with a [ReLU](#relu) activation.
- **Output:** 4 numbers, one per move (up, down, left, right). Each number is that move's **Q-value**: an estimate of how much reward the agent will collect from now on if it plays that move.
- **Value of a board:** its best Q-value among the valid moves (0 if the game is over). This is what the search uses.
- **Reward for each move:** log2 of every merged tile (merging into a 512 gives 9), plus 0.1 per empty cell.
  Merges move the game forward, and empty cells keep it alive.

### The search (expectimax)

Before each move, the agent:
1. Tries all 4 moves.
2. For each one, lists every possible new tile: each empty cell, with a 2 (90% chance) or a 4 (10% chance).
3. Scores each resulting board as `reward + 0.99 × its value`, where the value comes from the network, or, if the search goes deeper, from repeating steps 1-4 on that board.
4. Takes the probability-weighted average of these scores for each move, and plays the best move.

The **depth** is how many moves ahead the search looks before trusting the network.
Each extra level multiplies the number of boards to check by about 50, so the depth is **adaptive**: it depends on how dangerous the board is.

| Empty cells | Moves searched ahead |
|---|---|
| 5 or more | 2 (open board, little can go wrong) |
| 3-4 | 3 |
| 2 or fewer | 4 (one bad move can end the game) |

Most moves are played on open boards and stay cheap; the expensive searches happen on crowded boards, where games are lost.
Crowded boards also have few empty cells, so few possible new tiles, which keeps a deep search there affordable.

To make this fast enough in Python, `lookahead.py` precomputes the result of sliding every possible row (65,536 of them),
handles thousands of boards at once with numpy, and scores them with the network in a single batch (capped at 1,000 boards, so memory stays bounded).

### Training

`lookahead_v1` was trained for 3M moves (2M, then continued for 1M), with these settings (see `train.py`):

- **Playing with search:** during training, the moves are chosen by a 1-move search on top of the network, not by the network alone.
  Training games therefore go further, and the network learns from many more late-game boards — the crowded positions the search later relies on it to judge.
- **Replay buffer:** every (board, move, reward, next board) of the last 500k moves is stored. Learning starts after the first 10k moves.
- **Updates:** every 4 moves, it samples 128 stored moves and nudges their Q-values toward `reward + 0.99 × value of the next board`,
  with the [Adam](#adam) optimizer and a constant learning rate of 1e-4.
- **Double DQN:** the network picks the best next move, and a slow-moving copy of it (updated every 30k moves) scores that move,
  so the targets don't chase themselves and overestimate.
- **Action masking:** invalid moves (that don't change the board) are never chosen and never used as targets.
- **Exploration:** at first the moves are 100% random, then this drops to 1% over the first 25% of training.
- **Side effect:** the network is only trained on the moves the search actually plays. Its Q-values for other moves drift,
  so it plays badly on its own (~1,500 points), but its judgment of *how good a board is* keeps improving — and that's all the search uses.

Training with a 2-move search instead (`lookahead_v2`) made results worse: 61% vs 82% of games reaching 2048 with a 2-move search.


## Evaluation method

- `evaluate.py` plays 100 games with deterministic moves. Each game uses a fixed seed, so a model always plays the same games, and two models are compared on exactly the same games.
- Seeds 0-99 were used throughout the project to compare models and make decisions. Decisions tuned on the same games can fit those games by luck,
  so the final results are measured on **fresh seeds** (1000-1099), never used before.
- With 100 games, a percentage has an uncertainty of a few points (±4 points around 82%, ±1.5 around 98%), so small differences between models are noise.
- During training, `score_callback.py` evaluates the model every 100k moves, which gives the learning curves (`plot_runs.py`).


## Evolution of the model

The first versions (500k to 5M training moves, no masking, merge-points reward, -2 for blocked moves) never got past a 256 tile:
`watch_ai.py` showed the AI pressing keys in a circular motion, without any sense of what was going on. Then, step by step
(network alone, greedy play, 100 games on seeds 0-99, final evaluation during training):

| Model | Change | Mean score | Reached 1024 |
|---|---|---|---|
| first versions | — | ~1,000 | 0% |
| masked | Action masking: invalid moves can't be chosen | not recorded | |
| onehot | One-hot input, log2 reward | not recorded | |
| big_v2 | Bigger network (2 × 256) | 2,328 | 0% |
| big_v3 | Tuning (buffer, exploration, more eval games) | 3,713 | 2% |
| empty_v1 | +0.1 reward per empty cell | 4,886 | 1% |
| empty_v2 | Same, trained for 4M moves | 7,472 | 18% |
| double_v1 | Double DQN | 6,826 | 14% |
| nstep_v1 | 3-step returns (reverted: no gain) | 6,354 | 5% |
| eps_v1 | Lower final exploration (1%) | 5,478 | 9% |

Greedy play plateaued around 7,000-8,000 points. The breakthrough was the **search**: `empty_v2` with a 2-move search went from ~7,800 to ~23,000 points.
Then training *with* a 1-move search (`lookahead_v1`) made the network better at judging late-game boards, and the adaptive 2-to-4-move search took it to 98% of games reaching 2048.


## Project structure

- `game.py`: the game itself, in Python, with a Tkinter GUI. Run it by itself to play with the keyboard.
- `environment.py`: the Gymnasium environment (observations, rewards, game over), built on `game.py`.
- `masked_dqn.py`: Stable-Baselines3's DQN, extended with action masking, Double DQN, and the search (in training and in play).
- `lookahead.py`: the expectimax search.
- `train.py`: trains a new model or continues training an existing one.
- `score_callback.py`: evaluates the model during training, to get learning curves.
- `evaluate.py`: plays 100 games and prints the results.
- `watch_ai.py`: watch the model play live in the GUI.
- `plot_runs.py`: draws the learning curves of several runs (`runs.png`).
- `make_figures.py`: draws the summary charts in `figures/` from results already measured.
- `make_gif.py`: records one game as an animated GIF.
- `model_files.py`: where each model's files are saved (`models/<name>/`).


## How to use

Requirements: Python 3, `stable-baselines3`, `gymnasium`, `torch`, `numpy`, `matplotlib`, `tqdm`, `rich` (and Tkinter for the GUI).

```bash
# Train (lookahead_v1: train_lookahead_depth=1 in train.py, 2M moves, then 1M more)
python train.py --model lookahead_v1 --steps 2000000 --seed 0 --eval-lookahead
python train.py --model lookahead_v1 --steps 1000000 --continue-training --eval-lookahead

# Evaluate (--lookahead 4 = adaptive 2-to-4-move search, ~3 min per game on CPU)
python evaluate.py --model lookahead_v1 --lookahead 4 --seed-offset 1000 --games 100 --per-game

# Watch it play
python watch_ai.py --model lookahead_v1 --lookahead 4

# Record one game as a GIF (figures/ai_playing.gif)
python make_gif.py --model lookahead_v1 --lookahead 4 --seed 0

# Learning curves
python plot_runs.py empty_v2 lookahead_v1 --lookahead
```

`--lookahead` takes 0 (network alone), 1, 2, 3 (2 to 3 moves) or 4 (2 to 4 moves).
`--seed-offset` sets the first game's seed, `--per-game` shows progress and each game's result.


## Glossary

### Q-value
The network's estimate of the total reward the agent will collect from a board if it plays a given move, then keeps playing well.
Rewards further in the future count a bit less (× 0.99 per move, the discount factor *gamma*).

### ReLU
The function applied by each hidden neuron to its result: `ReLU(x) = max(0, x)`. It keeps positive numbers and turns negative ones into 0.
Without it, stacking layers would be pointless: a chain of linear layers is equivalent to a single linear layer, which can only represent very simple relationships
("each tile adds or removes a fixed amount of value"). Cutting at 0 lets each neuron act like a switch that only activates for certain patterns
(e.g. "big tile in the corner AND row next to it full"), and combining many such switches lets the network represent complex judgments of a board.

### Adam
The method used to adjust the network's weights during training. Each update, the error of the network's Q-values gives, for each weight, a direction
that would reduce that error (the gradient). Plain gradient descent moves every weight a fixed small step in that direction.
Adam improves on this by keeping, for each weight, a running average of its recent gradients (momentum: it keeps going in consistent directions and ignores noise)
and of their size (each weight gets its own step size: smaller for weights with large, noisy gradients, larger for weights with small ones).
This makes training faster and more stable, which matters in DQN because the targets themselves are noisy estimates.

### Expectimax
A search for games with chance: the agent's turns take the **max** (it picks its best move), the random tile's turns take the **expected** value (the probability-weighted average of what can appear).
