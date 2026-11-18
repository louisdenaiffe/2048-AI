import numpy as np
import torch

from environment import NUM_TILE_VALUES, EMPTY_CELL_BONUS
from game import Game2048


# Lookahead (expectimax search) on top of a trained Q-network, used by MaskedDQN.predict() when lookahead_depth > 0.
#
# Depth 1: for each valid move, list EVERY possible new tile (each empty cell, a 2 with 90% chance or a 4 with 10% chance),
# and score the move as the average, over those possibilities, of: reward + gamma * best Q-value of the resulting board.
# Depth 2: same, but instead of trusting the Q-values of those resulting boards, it searches one more move from each of them
# (the best of their own depth-1 scores). Each extra level multiplies the number of boards by about 50.
#
# Everything works on whole batches of boards at once (numpy), in "exponent" form: 0 = empty, 1 = tile 2, 2 = tile 4, 3 = tile 8...
# A Python loop over thousands of boards per move would be far too slow for depth 2.


def _build_row_table():
    # Sliding a row to the left only depends on its 4 values, and there are only 16^4 = 65536 possible rows (exponents 0 to 15).
    # So we compute the result of every possible row once, with the game's own merge_line(), and then just look rows up.
    rows = np.stack(np.meshgrid(*[np.arange(16)] * 4, indexing="ij"), axis=-1).reshape(-1, 4) # all 65536 rows of exponents
    results = np.zeros_like(rows)
    merge_rewards = np.zeros(len(rows), dtype=np.float32) # log2 of the merged tiles, like compute_reward() in environment.py
    for i, row in enumerate(rows):
        merged, merges = Game2048.merge_line(np.where(row > 0, 2 ** row, 0))
        results[i] = np.log2(np.maximum(merged, 1)).astype(np.int64)
        merge_rewards[i] = np.log2(merges).sum() if merges else 0.0
    return np.minimum(results, NUM_TILE_VALUES - 1), merge_rewards
    # (np.minimum only matters for merging two 32768 tiles, which never happens in practice.)


_ROW_TABLE = None # built the first time it's needed (takes about a second)


def _row_table():
    global _ROW_TABLE
    if _ROW_TABLE is None:
        _ROW_TABLE = _build_row_table()
    return _ROW_TABLE


# How each move is turned into a "slide left" and back (same trick as preview_move() in game.py).
# Actions: 0 up, 1 down, 2 left, 3 right. grids has shape (N, 4 rows, 4 columns).
_ORIENT = {
    0: (lambda g: g.transpose(0, 2, 1), lambda g: g.transpose(0, 2, 1)), # up: columns become rows
    1: (lambda g: g.transpose(0, 2, 1)[:, :, ::-1], lambda g: g[:, :, ::-1].transpose(0, 2, 1)), # down: columns, reversed
    2: (lambda g: g, lambda g: g), # left: nothing to do
    3: (lambda g: g[:, :, ::-1], lambda g: g[:, :, ::-1]), # right: rows reversed
}


def all_moves(boards):
    # Plays the 4 moves on every board. boards: (N, 4, 4) exponents.
    # Returns the boards after each move (N, 4 moves, 4, 4), the merge reward of each move (N, 4), and which moves are valid (N, 4).
    results, merge_rewards = _row_table()
    after = np.empty((len(boards), 4, 4, 4), dtype=boards.dtype)
    rewards = np.empty((len(boards), 4), dtype=np.float32)
    for action, (to_left, back) in _ORIENT.items():
        rows = to_left(boards)
        index = rows[..., 0] * 4096 + rows[..., 1] * 256 + rows[..., 2] * 16 + rows[..., 3] # each row's number in the table
        after[:, action] = back(results[index])
        rewards[:, action] = merge_rewards[index].sum(axis=1)
    valid = (after != boards[:, None]).any(axis=(2, 3)) # a move is valid if it changes the board
    return after, rewards, valid


def add_new_tiles(boards):
    # Lists every possible new tile for every board: each empty cell, with a 2 (90%) or a 4 (10%).
    # Returns the new boards (K, 4, 4), which board each one comes from (K,), and its probability (K,).
    flat = boards.reshape(len(boards), 16)
    empty = flat == 0
    board_index, cell = np.nonzero(empty)
    empty_count = empty.sum(axis=1)
    children, parents, probabilities = [], [], []
    for tile_exponent, tile_probability in ((1, 0.9), (2, 0.1)): # exponent 1 = tile 2, exponent 2 = tile 4
        child = flat[board_index].copy()
        child[np.arange(len(cell)), cell] = tile_exponent
        children.append(child)
        parents.append(board_index)
        probabilities.append(tile_probability / empty_count[board_index]) # each empty cell is equally likely
    return np.concatenate(children).reshape(-1, 4, 4), np.concatenate(parents), np.concatenate(probabilities)


def _best_q_values(model, boards):
    # Best valid Q-value of each board (0 if the game is over there), all in one batch through the network.
    if len(boards) == 0:
        return np.zeros(0, dtype=np.float32)
    from masked_dqn import valid_action_mask # imported here to avoid a circular import (masked_dqn imports this file)
    one_hot = np.eye(NUM_TILE_VALUES, dtype=np.float32)[boards.reshape(len(boards), 16)].reshape(len(boards), -1) # see encode_board()
    observations = torch.as_tensor(one_hot, device=model.device)
    with torch.no_grad():
        q_values = model.q_net(observations)
    mask = valid_action_mask(observations)
    q_values[~mask] = -torch.inf
    best = q_values.max(dim=1).values
    return torch.where(mask.any(dim=1), best, torch.zeros_like(best)).cpu().numpy()


def move_values(model, boards, depth):
    # Expected value of each of the 4 moves on each board: (N, 4), -inf for blocked moves.
    after, merge_rewards, valid = all_moves(boards)
    board_index, action = np.nonzero(valid) # one entry per valid (board, move) pair
    moved = after[board_index, action]

    children, parent, probability = add_new_tiles(moved)
    reward = merge_rewards[board_index, action][parent] + EMPTY_CELL_BONUS * (children == 0).reshape(len(children), 16).sum(axis=1)
    # Same reward as compute_reward() in environment.py: log2 of the merged tiles + bonus per empty cell after the new tile.

    if depth == 1:
        future = _best_q_values(model, children) # trust the network from here
    else:
        child_values = move_values(model, children, depth - 1) # search one more move from each resulting board
        future = np.where(np.isfinite(child_values).any(axis=1), child_values.max(axis=1), 0.0) # game over: no future, so 0

    contributions = probability * (reward + model.gamma * future)
    expected = np.bincount(parent, weights=contributions, minlength=len(moved)) # adds up the possibilities of each (board, move) pair
    values = np.full((len(boards), 4), -np.inf)
    values[board_index, action] = expected
    return values


def lookahead_action(model, observation, depth):
    # The move chosen by a lookahead of the given depth, for one one-hot encoded board (see encode_board()).
    board = observation.reshape(16, NUM_TILE_VALUES).argmax(axis=1).reshape(1, 4, 4) # one-hot -> exponents
    return int(move_values(model, board, depth)[0].argmax())
