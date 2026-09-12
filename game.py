import numpy as np
import tkinter as tk
from tkinter import messagebox


class Game2048:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board.fill(0)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.get_board()

    def add_random_tile(self):
        empty_spaces = np.argwhere(self.board == 0)

        if len(empty_spaces) == 0:
            return False

        position = self.rng.integers(len(empty_spaces))
        row, column = empty_spaces[position]
        self.board[row, column] = 4 if self.rng.random() < 0.1 else 2
        return True

    @staticmethod
    def merge_line(line):
        numbers = line[line != 0].tolist()
        merged_numbers = []
        points_gained = 0
        index = 0

        while index < len(numbers):
            if (
                index + 1 < len(numbers)
                and numbers[index] == numbers[index + 1]
            ):
                new_value = numbers[index] * 2
                merged_numbers.append(new_value)
                points_gained += new_value
                index += 2
            else:
                merged_numbers.append(numbers[index])
                index += 1

        merged_numbers += [0] * (4 - len(merged_numbers))

        return (
            np.array(merged_numbers, dtype=np.int32),
            points_gained,
        )

    def move_left(self):
        old_board = self.board.copy()
        points_gained = 0

        for row in range(4):
            merged, points = self.merge_line(self.board[row])
            self.board[row] = merged
            points_gained += points

        return self.finish_move(old_board, points_gained)

    def move_right(self):
        old_board = self.board.copy()
        points_gained = 0

        for row in range(4):
            reversed_row = self.board[row][::-1]
            merged, points = self.merge_line(reversed_row)
            self.board[row] = merged[::-1]
            points_gained += points

        return self.finish_move(old_board, points_gained)

    def move_up(self):
        old_board = self.board.copy()
        points_gained = 0

        for column in range(4):
            merged, points = self.merge_line(self.board[:, column])
            self.board[:, column] = merged
            points_gained += points

        return self.finish_move(old_board, points_gained)

    def move_down(self):
        old_board = self.board.copy()
        points_gained = 0

        for column in range(4):
            reversed_column = self.board[:, column][::-1]
            merged, points = self.merge_line(reversed_column)
            self.board[:, column] = merged[::-1]
            points_gained += points

        return self.finish_move(old_board, points_gained)

    def finish_move(self, old_board, points_gained):
        changed = not np.array_equal(self.board, old_board)

        if changed:
            self.score += points_gained
            self.add_random_tile()

        return points_gained, changed

    def apply_move(self, action):
        moves = {
            self.ACTION_UP: self.move_up,
            self.ACTION_DOWN: self.move_down,
            self.ACTION_LEFT: self.move_left,
            self.ACTION_RIGHT: self.move_right,
        }

        if action not in moves:
            raise ValueError("Action must be 0, 1, 2, or 3")

        return moves[action]()

    def get_valid_actions(self):
        """Return actions that would change the board."""
        valid_actions = []
        original_board = self.board.copy()

        for action in range(4):
            test_game = Game2048.__new__(Game2048)
            test_game.board = original_board.copy()
            test_game.score = 0
            test_game.rng = np.random.default_rng()

            # Do not call finish_move here: checking whether a move is valid
            # should never add a random tile or otherwise mutate game state.
            points_gained = 0
            if action in (self.ACTION_LEFT, self.ACTION_RIGHT):
                for row in range(4):
                    line = test_game.board[row]
                    if action == self.ACTION_RIGHT:
                        line = line[::-1]
                    merged, points = self.merge_line(line)
                    test_game.board[row] = (
                        merged[::-1]
                        if action == self.ACTION_RIGHT
                        else merged
                    )
                    points_gained += points
            else:
                for column in range(4):
                    line = test_game.board[:, column]
                    if action == self.ACTION_DOWN:
                        line = line[::-1]
                    merged, points = self.merge_line(line)
                    test_game.board[:, column] = (
                        merged[::-1]
                        if action == self.ACTION_DOWN
                        else merged
                    )

            changed = not np.array_equal(test_game.board, original_board)

            if changed:
                valid_actions.append(action)

        return valid_actions

    def is_game_over(self):
        if np.any(self.board == 0):
            return False

        if np.any(self.board[:, :-1] == self.board[:, 1:]):
            return False

        if np.any(self.board[:-1, :] == self.board[1:, :]):
            return False

        return True

    def get_board(self):
        return self.board.copy()


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

    KEY_ACTIONS = {
        "up": Game2048.ACTION_UP,
        "down": Game2048.ACTION_DOWN,
        "left": Game2048.ACTION_LEFT,
        "right": Game2048.ACTION_RIGHT,
        "w": Game2048.ACTION_UP,
        "s": Game2048.ACTION_DOWN,
        "a": Game2048.ACTION_LEFT,
        "d": Game2048.ACTION_RIGHT,
    }

    def __init__(self, root, game):
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

        tk.Label(
            root,
            text="Arrow keys or WASD to move",
            font=("Helvetica", 12),
            bg="#faf8ef",
            fg="#776e65",
        ).pack(pady=5)

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

        # bind_all keeps controls working even after the New Game button has
        # taken keyboard focus. Normalizing keysyms also supports uppercase
        # WASD (Caps Lock/Shift) consistently across platforms.
        self.root.bind_all("<KeyPress>", self.handle_key)
        self.root.focus_set()
        self.update_display()

    def handle_key(self, event):
        key = event.keysym.lower()

        if key in self.KEY_ACTIONS:
            self.make_move(self.KEY_ACTIONS[key])
            return "break"

    def make_move(self, action):
        # Ignore queued key-repeat events while a move/dialog is being handled.
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
    root = tk.Tk()
    game = Game2048(seed=42)
    gui = Game2048GUI(root, game)

    # To display an ML agent, uncomment these lines:
    #
    # def choose_action(board):
    #     valid_actions = game.get_valid_actions()
    #     return int(game.rng.choice(valid_actions))
    #
    # gui.run_agent(choose_action, delay=150)

    root.mainloop()
