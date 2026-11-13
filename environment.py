import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import Game2048


NUM_TILE_VALUES = 16 # possible values per cell: empty, 2, 4, 8, ..., 32768 (exponents 0 to 15)


def encode_board(board):
    # Converts the 4x4 board into the 256 numbers the model receives (one-hot encoding):
    # each of the 16 cells becomes 16 yes/no values, with a single 1 at the position of its tile's exponent.
    exponents = np.log2(np.maximum(board, 1)).astype(np.int64).ravel()
    # np.maximum transforms 0s into 1s to avoid log2(0), so empty cells get exponent 0.
    # np.log2() converts tiles into their exponents: 2 -> 1, 4 -> 2, 8 -> 3, etc...
    # .astype(np.int64) makes them whole numbers, so they can be used as indices. .ravel() flattens the 4x4 grid into 16 values, row by row.
    one_hot = np.eye(NUM_TILE_VALUES, dtype=np.float32)[exponents]
    # np.eye(16) is the identity matrix: row k has a single 1, in position k, which is exactly the one-hot vector of exponent k.
    # Indexing it with the 16 exponents picks one row per cell, giving a (16 cells, 16 values) array.
    return one_hot.ravel()
    # Flattens it into 256 values: the 16 values of cell 0, then the 16 values of cell 1, etc...


class Game2048Env(gym.Env): # Defines the env by inheriting from the Gym's base env class, gym.Env
    metadata = {"render_modes": ["human"]} # Declares that the env supports rendering for a human to watch. (doesn't create a GUI tho)

    def __init__(self, max_moves=10_000): # !)k max moves per episode
        super().__init__() # Calls the parent's class initializer (so gym.Env's)
        self.game = Game2048() # Creates the game (an instance of the game class).
        self.max_moves = max_moves
        self.moves = 0
        self.action_space = spaces.Discrete(4) # Four possible action numbers: 0, 1, 2, 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(16 * NUM_TILE_VALUES,), dtype=np.float32)
        # Here, this defines the observation space (the EXPECTED data format):
        # shape(256,) is a one-dimensional array of 16 cells x 16 possible values = 256 values
        # low=0, high=1 means each encoded value is between 0 and 1 (here, always exactly 0 or 1)
        # dtype=np,float32 each value is a 32-bit floating-point number


    def get_observation(self):
        return encode_board(self.game.board)


    def get_info(self):
        return {
            "score": self.game.score,
            "largest_tile": int(self.game.board.max()),
            "moves": self.moves
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # super() refers to the parent class's (Gym.env) functions/methods
        # Resets the Gymnasium environment with some randomness directed by the seed.
        self.game.reset(seed=seed) # careful: this is the game's reset, not Gymnasium. It clears the board, etc...
        self.moves = 0
        return self.get_observation(), self.get_info() # useful for evaluation


    def step(self, action): # runs one timestep of the env, one action.
        merges_extended, _ = self.game.apply_move(int(action)) # This is what the game apply_move() function returns based on the action
        # apply_move() runs preview_move(), which simulates the move without changing the board.
        # If it sees that it will change the board, it applies that move. (The second value it returns, "changed", isn't needed here:
        # MaskedDQN never chooses a blocked move, and even if one arrived, apply_move() would simply do nothing.)
        self.moves += 1
        reward = float(np.log2(merges_extended).sum())
        # reward is downscaled using log2, which means the more points the AI gets, the more it is rewarded, encouraging it to get more points.
        terminated = self.game.is_game_over() # checks if the game is over (grid covered in tiles, no more moves possible)
        truncated = self.moves >= self.max_moves # if moves is more than 10k, then termination of the env
        # Safety net only: with masking, every move changes the board and games always end naturally.
        info = self.get_info()
        return self.get_observation(), reward, terminated, truncated, info


    def render(self):
        print(self.game.board)
        print(f"Score: {self.game.score}")
        print(f"Moves: {self.moves}")
