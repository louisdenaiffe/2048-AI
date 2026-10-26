import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import Game2048


def encode_board(board):
    # Converts the 4x4 board into the 16 numbers the model receives
    return np.log2(np.maximum(board, 1)).astype(np.float32).ravel()
    # Here: np.maximum transforms 0s into 1s to avoid log2(0)
    # np.log2() converts tiles into their exponents. So 1 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, etc... This keeps inputs on a smaller scale.
    # .astype(np.float32) converts the numbers into the numeric type awaited by the env.
    # .ravel() flattens the 4x4 grid into an array, reading row by row.


class Game2048Env(gym.Env): # Defines the env by inheriting from the Gym's base env class, gym.Env
    metadata = {"render_modes": ["human"]} # Declares that the env supports rendering for a human to watch. (doesn't create a GUI tho)

    def __init__(self, max_moves=10_000): # !)k max moves per episode
        super().__init__() # Calls the parent's class initializer (so gym.Env's)
        self.game = Game2048() # Creates the game (an instance of the game class).
        self.max_moves = max_moves
        self.moves = 0
        self.action_space = spaces.Discrete(4) # Four possible action numbers: 0, 1, 2, 3
        self.observation_space = spaces.Box(low=0, high=31, shape=(16,), dtype=np.float32)
        # Here, this defines the observation space (the EXPECTED data format):
        # shape(16,) is a one-dimensional array of 16 values
        # low=0, high=31 means each encoded value must be between 0 and 31
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
        points_gained, changed = self.game.apply_move(int(action)) # This is what the game apply_move() function returns based on the action
        # apply_move() runs preview_move(), which simulates the move without changing the board. 
        # If it sees that it will change the board, it applies that move and outputs changed=True.
        self.moves += 1
        reward = float(points_gained) 
        # reward is set to points gained, which means the more points the AI gets, the more it is rewarded, encouraging it to get more points.
        blocked_move = False
        if not changed: # if the preview of that move reveals that the board hasn'tchanged, then reward is -2.
            reward -= 2.0
            blocked_move = True
        terminated = self.game.is_game_over() # checks if the game is over (grid covered in tiles, no more moves possible)
        truncated = self.moves >= self.max_moves # if moves is more than 10k, then termination of the env
        return self.get_observation(), reward, terminated, truncated, blocked_move, self.get_info()

    def render(self):
        print(self.game.board)
        print(f"Score: {self.game.score}")
        print(f"Moves: {self.moves}")
