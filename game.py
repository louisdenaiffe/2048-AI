import numpy as np


# Class with all the basic functions to make the game work: reset(), add_random_tile(), merge_line(), 
# preview_move(), apply_move(), get_valid_actions(), is_game_over(), get_board()


class Game2048:
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = range(4)
    ACTION_NAMES = ("Up", "Down", "Left", "Right")

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        # Creates a new null matrix with 4 rows and 4 columns
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # If there is a seed, then generate a new random number generator (rng) using that seed.
        # Later on, this rng can be used to create random numbers, deterministically, so different models can be tested on identical games.
        self.board.fill(0) # Clears the existing matrix (board)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        # Sets score to 0 and adds the two initial tiles to the cleared board.
        return self.get_board() # returns a copy of the board

    def add_random_tile(self):
        empty = np.flatnonzero(self.board == 0)
        # self.board == 0 builds a 4x4 grid of True and Falses, with True where a cell is empty.
        # Then np.flatnonzero() returns the indices of these empty cells, as if the board were a row of 16 cells.
        if not empty.size: # if there are no empty cells, then return False, because there is no space.
            return False
        position = empty[self.rng.integers(empty.size)]
        # self.rng.integers(n) chooses a random whole number between 0 and n-1. that number is an indice of an indice of an empty cell. (not a typo)
        # as such, position takes the indice of an empty cell in the board.
        self.board.flat[position] = 4 if self.rng.random() < 0.1 else 2
        # In 90% of cases, that empty cell becomes a 2, and in 10% of cases, it becomes a 4.
        return True

    @staticmethod
    def merge_line(line): # careful, input here is not the board but an individual row
        tiles = line[line != 0].tolist() # transforms it into a list, without the null elements. (so np.array([3, 0, 3, 4]) -> [3, 3, 4])
        merged, points, i = [], 0, 0
        # The whole merging math, boring
        while i < len(tiles):
            value = tiles[i]
            if i + 1 < len(tiles) and value == tiles[i + 1]:
                value *= 2
                points += value
                i += 1
            merged.append(value)
            i += 1
        return np.array(merged + [0] * (4 - len(merged)), dtype=np.int32), points
        # returns the row, merged, as well as the points gained from merging.
        # IMPORTANT: why only left? what if the agent decided on right, up or down? W
        # ell, preview_move() flips the board horizontally and vertically! thats it

    def preview_move(self, action):
        if action not in range(4):
            raise ValueError("Action must be 0, 1, 2, or 3")
        board = self.board.copy() # we copy the board so as to not change it yet (this is a preview)
        rows = board.T if action in (self.ACTION_UP, self.ACTION_DOWN) else board
        # Orients the board so every action becomes a left merge (board.T is the transpose of the board)
        # These are writable views: updating a row also updates the copied board.
        if action in (self.ACTION_DOWN, self.ACTION_RIGHT):
            rows = rows[:, ::-1]
        points = 0
        for row in rows:
            merged, gained = self.merge_line(row) # merges each row individually
            row[:] = merged
            points += gained
        return board, points, not np.array_equal(board, self.board)
        # returns what the board looks like after that previewed move, the points this move would earn, and whether the board has changed.

    def apply_move(self, action):
        board, points, changed = self.preview_move(action)
        if changed:
            self.board[:] = board
            self.score += points
            self.add_random_tile()
        # If the board has changed after previewing the move, then apply that move.
        # Then add another random tile to the board.
        return points, changed

    def get_valid_actions(self):
        return [action for action in range(4) if self.preview_move(action)[2]]

    def is_game_over(self):
        return not (
            np.any(self.board == 0)
            or np.any(self.board[:, :-1] == self.board[:, 1:])
            or np.any(self.board[:-1, :] == self.board[1:, :])
        )

    def get_board(self):
        return self.board.copy()


# Class for rendering the game visually. 
# Function include: make_move(), show_game_over(), update_display(), reset(), stop_agent(), run_agent() (and step_agent() inside)

class Game2048GUI:
    CELL_SIZE = 100

    TILE_COLORS = {
        0: ("#cdc1b4", "#776e65"),
        2: ("#eee4da", "#776e65"),
        4: ("#ede0c8", "#776e65"),
        8: ("#f2b179", "#f9f6f2"),
        16: ("#f59563", "#f9f6f2"),
        32: ("#f67c5f", "#f9f6f2"),
        64: ("#f65e3b", "#f9f6f2"),
        128: ("#edcf72", "#f9f6f2"),
        256: ("#edcc61", "#f9f6f2"),
        512: ("#edc850", "#f9f6f2"),
        1024: ("#edc53f", "#f9f6f2"),
        2048: ("#edc22e", "#f9f6f2"),
    }

    def __init__(self, root, game):
        import tkinter as tk

        self.root = root
        self.game = game
        self.cells = []
        self.game_over_displayed = False
        self.move_in_progress = False
        self.agent_job = None

        self.root.title("2048")
        # Leave enough room for the entire fixed-size board and controls.
        self.root.geometry("520x720")
        self.root.resizable(False, False)
        self.root.configure(bg="#faf8ef")

        tk.Label(
            root,
            text="2048",
            font=("Helvetica", 44, "bold"),
            bg="#faf8ef",
            fg="#776e65",
        ).pack(pady=(20, 5))

        self.score_label = tk.Label(
            root,
            text="Score: 0",
            font=("Helvetica", 18, "bold"),
            bg="#faf8ef",
            fg="#776e65",
        )
        self.score_label.pack(pady=5)

        board_frame = tk.Frame(
            root,
            bg="#bbada0",
            padx=10,
            pady=10,
        )
        board_frame.pack(pady=15)

        for row in range(4):
            cell_row = []

            for column in range(4):
                # Label width/height are measured in characters, not pixels.
                # Using them for the tile itself made every font-size change
                # resize the grid, so tiles appeared to jump or disappear.
                cell_container = tk.Frame(
                    board_frame,
                    width=self.CELL_SIZE,
                    height=self.CELL_SIZE,
                    bg="#cdc1b4",
                )
                cell_container.grid(
                    row=row,
                    column=column,
                    padx=5,
                    pady=5,
                )
                cell_container.grid_propagate(False)

                cell = tk.Label(
                    cell_container,
                    text="",
                    font=("Helvetica", 24, "bold"),
                    relief="flat",
                    bg="#cdc1b4",
                )
                cell.place(x=0, y=0, relwidth=1, relheight=1)
                cell_row.append(cell)

            self.cells.append(cell_row)

        tk.Button(
            root,
            text="New Game",
            command=self.reset,
            font=("Helvetica", 13, "bold"),
            bg="#8f7a66",
            fg="white",
            activebackground="#9f8b77",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=8,
        ).pack(pady=10)

        self.update_display()

    def make_move(self, action):
        # Prevent re-entrant moves while a move/dialog is being handled.
        if self.move_in_progress or self.game.is_game_over():
            self.show_game_over()
            return 0, False

        self.move_in_progress = True
        try:
            points, changed = self.game.apply_move(action)
            self.update_display()
            self.show_game_over()
            return points, changed
        finally:
            self.move_in_progress = False

    def show_game_over(self):
        from tkinter import messagebox

        if self.game.is_game_over() and not self.game_over_displayed:
            self.game_over_displayed = True
            messagebox.showinfo(
                "Game Over",
                f"Game over!\nFinal score: {self.game.score}",
            )

    def update_display(self):
        board = self.game.get_board()
        self.score_label.config(text=f"Score: {self.game.score}")

        for row in range(4):
            for column in range(4):
                value = int(board[row, column])
                background, foreground = self.TILE_COLORS.get(
                    value,
                    ("#3c3a32", "#f9f6f2"),
                )

                font_size = 24
                if value >= 1024:
                    font_size = 18
                elif value >= 128:
                    font_size = 21

                self.cells[row][column].config(
                    text=str(value) if value != 0 else "",
                    bg=background,
                    fg=foreground,
                    font=("Helvetica", font_size, "bold"),
                )

    def reset(self, seed=None):
        self.stop_agent()
        self.game.reset(seed)
        self.game_over_displayed = False
        self.move_in_progress = False
        self.update_display()

    def stop_agent(self):
        if self.agent_job is not None:
            self.root.after_cancel(self.agent_job)
            self.agent_job = None

    def run_agent(self, choose_action, delay=150):
        """
        Visually run an ML agent.

        choose_action receives a copy of the board and must return:
            0 = up
            1 = down
            2 = left
            3 = right
        """

        def agent_step():
            self.agent_job = None
            if self.game.is_game_over():
                self.update_display()
                self.show_game_over()
                return

            action = choose_action(self.game.get_board())
            self.make_move(action)
            self.agent_job = self.root.after(delay, agent_step)

        self.stop_agent()
        self.agent_job = self.root.after(delay, agent_step)


if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()
    game = Game2048(seed=42)
    gui = Game2048GUI(root, game)
    root.mainloop()
