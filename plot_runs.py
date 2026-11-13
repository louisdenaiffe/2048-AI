import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


from model_files import model_files


# Draws the learning curves saved by score_callback.py (models/<name>/eval.csv), so training runs can be compared.
# Usage: python plot_runs.py onehot_v2 big_v2 --out runs.png


# Reference scores of strategies that don't learn, measured with evaluate.py's 100 games (seeds 0-99).
BASELINES = {
    "Corner rule (down > left > right > up)": {"mean_score": 2580, "reached_512": 8.0},
    "Random valid moves": {"mean_score": 1074, "reached_512": 0.0},
}

# Colors: one per run, always in this order (run 1 is always blue, run 2 always orange...), checked to stay distinguishable for colorblind readers.
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# The two things plotted, one panel each (two separate charts rather than two scales on one chart, which is hard to read).
PANELS = [
    ("mean_score", "Mean score", "{:.0f}"),
    ("reached_512", "Games reaching 512 (%)", "{:.0f}%"),
]


def read_run(csv_path):
    # Reads one .eval.csv file into a dictionary of columns: {"timesteps": [...], "mean_score": [...], ...}
    with open(csv_path, newline="") as file:
        rows = list(csv.DictReader(file))
    return {column: [float(row[column]) for row in rows] for column in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="One or more model names (reads models/<name>/eval.csv)") # nargs="+" means "one or more values"
    parser.add_argument("--out", default="runs.png", help="Where to save the image")
    parser.add_argument("--show", action="store_true", help="Also open the chart in a window")
    args = parser.parse_args()

    if len(args.models) > len(RUN_COLORS):
        parser.error(f"At most {len(RUN_COLORS)} runs per chart, so each one keeps a distinct color.")

    runs = {}
    for name in args.models:
        csv_path = model_files(name).eval_csv
        if not csv_path.exists():
            parser.error(f"No evaluation results at {csv_path} (models trained before score_callback.py existed don't have any)")
        runs[Path(name).name] = read_run(csv_path) # Path(name).name keeps only the last part, in case a path was given

    figure, axes = plt.subplots(1, len(PANELS), figsize=(13, 5), facecolor=SURFACE)
    # plt.subplots(1, 2) creates one figure with 1 row of 2 charts ("axes" in matplotlib's vocabulary).

    for ax, (column, title, value_format) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)

        # Baselines first, so they sit behind the runs: thin dashed gray lines, labelled on the left.
        for name, values in BASELINES.items():
            ax.axhline(values[column], color=MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
            ax.annotate(name, xy=(0, values[column]), xycoords=("axes fraction", "data"), xytext=(4, 3),
                        textcoords="offset points", fontsize=8, color=MUTED, va="bottom")

        # One line per run, with a dot at each evaluation.
        for (name, data), color in zip(runs.items(), RUN_COLORS):
            steps_in_millions = [step / 1_000_000 for step in data["timesteps"]]
            ax.plot(steps_in_millions, data[column], color=color, linewidth=2, marker="o", markersize=4, label=name,
                    zorder=3, clip_on=False) # clip_on=False: dots exactly on 0 aren't cut in half by the bottom of the chart

        # Each run's name + last value, in a column to the right of all the lines, spread out so they never overlap.
        top = max([max(data[column]) for data in runs.values()] + [values[column] for values in BASELINES.values()])
        ax.set_ylim(0, top * 1.1 if top > 0 else 1)
        min_gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06 # minimum vertical distance between two labels
        labels = sorted((data[column][-1], name) for name, data in runs.items()) # sorted from lowest to highest last value
        label_positions, previous = [], None
        for value, name in labels:
            position = value if previous is None else max(value, previous + min_gap) # pushes a label up if too close to the one below
            label_positions.append((position, value, name))
            previous = position
        for position, value, name in label_positions:
            ax.annotate(f"{name}  {value_format.format(value)}", xy=(1.01, position), xycoords=("axes fraction", "data"),
                        fontsize=8, color=INK, va="center", annotation_clip=False)

        # Discreet axes and grid, so the data stands out.
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_xlabel("Training steps (millions)", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.margins(x=0.03)

    # One legend for the whole figure, above the charts (the colors are the same in both panels).
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.93)) # leaves space at the top for the legend
    figure.subplots_adjust(right=0.88, wspace=0.35) # leaves room on the right of each chart for the run labels

    figure.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Saved chart to {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
