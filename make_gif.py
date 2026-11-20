import argparse
import math
import time

from PIL import Image, ImageDraw, ImageFont

from environment import Game2048Env
from game import Game2048GUI
from masked_dqn import MaskedDQN
from model_files import find_model_to_load


# Plays one game with a trained model and saves it as an animated GIF (e.g. for the README).
# Usage: python make_gif.py --model lookahead_v1 --lookahead 4 --seed 0 --out figures/ai_playing.gif
# The game is played first (about 3 minutes with --lookahead 4), then drawn with the same colors as game.py's GUI.
# Same seed + same model + same lookahead = same game, so the GIF can always be made again.


BACKGROUND, GRID, TEXT = "#faf8ef", "#bbada0", "#776e65"
TILE, GAP, HEADER = 80, 10, 56 # tile size, space between tiles, height of the score line above the board (pixels)
BOARD = 4 * TILE + 5 * GAP # 370 x 370 board


def load_font(size):
    # A bold font for the numbers; Pillow's built-in font if DejaVu isn't installed.
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


FONTS = {size: load_font(size) for size in (30, 26, 21, 17, 18)}


def draw_frame(board, score):
    # One GIF frame: a score line, then the 4x4 board.
    image = Image.new("RGB", (BOARD, BOARD + HEADER), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((GAP, HEADER / 2), f"Score {score:,}", font=FONTS[18], fill=TEXT, anchor="lm") # left-aligned
    draw.text((BOARD - GAP, HEADER / 2), f"Best tile {int(board.max())}", font=FONTS[18], fill=TEXT, anchor="rm") # right-aligned
    draw.rounded_rectangle((0, HEADER, BOARD, HEADER + BOARD), radius=8, fill=GRID)
    for row in range(4):
        for column in range(4):
            value = int(board[row, column])
            background, foreground = Game2048GUI.TILE_COLORS.get(value, ("#3c3a32", "#f9f6f2")) # same colors as the GUI
            x = GAP + column * (TILE + GAP)
            y = HEADER + GAP + row * (TILE + GAP)
            draw.rounded_rectangle((x, y, x + TILE, y + TILE), radius=5, fill=background)
            if value:
                size = 30 if value < 128 else 26 if value < 1024 else 21 if value < 16384 else 17 # smaller font for longer numbers
                draw.text((x + TILE / 2, y + TILE / 2), str(value), font=FONTS[size], fill=foreground, anchor="mm")
    return image


def play_game(model, seed):
    # Plays one game and returns every position: (board, score, move number), from the start to the end.
    env = Game2048Env()
    observation, info = env.reset(seed=seed)
    positions = [(env.game.board.copy(), 0, 0)]
    start, finished = time.time(), False
    while not finished:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        positions.append((env.game.board.copy(), info["score"], info["moves"]))
        finished = terminated or truncated
        if info["moves"] % 100 == 0:
            print(f"\r  move {info['moves']}, score {info['score']:,}, best tile {info['largest_tile']} "
                  f"[{time.time() - start:.0f}s]", end="", flush=True)
    print(f"\rGame over after {info['moves']} moves: score {info['score']:,}, best tile {info['largest_tile']} "
          f"[{time.time() - start:.0f}s]")
    env.close()
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lookahead_v1", help="A model name (models/<name>/model.zip) or a model file")
    parser.add_argument("--lookahead", type=int, default=4, choices=[0, 1, 2, 3, 4], help="Search depth, as in evaluate.py")
    parser.add_argument("--seed", type=int, default=0, help="Which game to play (seed 0 reaches 4096 with lookahead_v1 --lookahead 4)")
    parser.add_argument("--out", default="figures/ai_playing.gif")
    parser.add_argument("--frames", type=int, default=300,
                        help="About how many frames the fast part of the GIF has (more = smoother but bigger file)")
    parser.add_argument("--ending", type=int, default=40, help="The last N moves are all shown, at a slower speed")
    args = parser.parse_args()

    model = MaskedDQN.load(find_model_to_load(args.model), device="cpu")
    model.lookahead_depth = args.lookahead
    print(f"Playing seed {args.seed} with {args.model}, lookahead {args.lookahead}...")
    positions = play_game(model, args.seed)

    # A full game has thousands of moves: the GIF shows every k-th move quickly, then every move of the ending more slowly.
    fast_part = positions[:max(len(positions) - args.ending, 1)]
    step = max(1, math.ceil(len(fast_part) / args.frames))
    shown = fast_part[::step] + positions[len(fast_part):]
    durations = [60] * len(fast_part[::step]) + [150] * (len(shown) - len(fast_part[::step])) # milliseconds per frame
    durations[0] = 1000 # the starting board stays 1 s
    durations[-1] = 4000 # the final board stays 4 s

    print(f"Drawing {len(shown)} frames (every {step} moves, then the last {len(shown) - len(fast_part[::step])} one by one)...")
    frames = [draw_frame(board, score) for board, score, _ in shown]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    # loop=0: the GIF repeats forever. Pillow converts each frame to a 256-color palette, plenty for these flat colors.
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
