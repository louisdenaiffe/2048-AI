import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import Game2048


def encode_board(board):
    """Flatten log2 tiles, with zero for empty cells."""
    return np.log2(np.maximum(board, 1)).astype(np.float32).ravel()


class Game2048Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_moves=10_000):
        super().__init__()
        self.game = Game2048()
        self.max_moves = max_moves
        self.moves = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=31, shape=(16,), dtype=np.float32)

    def get_observation(self):
        return encode_board(self.game.board)

    def get_info(self):
        return {
            "score": self.game.score,
            "largest_tile": int(self.game.board.max()),
            "moves": self.moves
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self.moves = 0
        return self.get_observation(), self.get_info()

    def step(self, action):
        points_gained, changed = self.game.apply_move(int(action))
        self.moves += 1
        reward = float(points_gained)
        if not changed:
            reward -= 2.0
        terminated = self.game.is_game_over()
        truncated = self.moves >= self.max_moves
        return self.get_observation(), reward, terminated, truncated, self.get_info()

    def render(self):
        print(self.game.board)
        print(f"Score: {self.game.score}")
        print(f"Moves: {self.moves}")
