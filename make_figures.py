import matplotlib.pyplot as plt

from plot_runs import RUN_COLORS, INK, MUTED, GRID, AXIS, SURFACE


# Draws the summary charts of the write-up (figures/), from results ALREADY measured: nothing is re-evaluated,
# so the code changes since then don't affect them. The learning-curve chart comes from plot_runs.py instead:
#   python plot_runs.py big_v3 empty_v1 empty_v2 double_v1 nstep_v1 --out figures/1_plateau.png
# Unless noted, results are evaluate.py on seeds 0-99 (100 games), copied from the journal.

NETWORK_ALONE, WITH_SEARCH = RUN_COLORS[0], RUN_COLORS[1]

# (label, mean score, played with search?)
PROGRESSION = [
    ("smoke: first model", 914, False),
    ("+ masking when playing", 1374, False),
    ("masked_v1: masking in training too", 2507, False),
    ("onehot_v1: one-hot input, log2 reward", 2572, False),
    ("big_v5: bigger network + tuning", 3457, False),
    ("empty_v1: empty-cell bonus", 4803, False),
    ("empty_v2: same, 4M steps", 7413, False),
    ("empty_v2 + 2-move search", 22736, True),
    ("lookahead_v1 (search in training) + 2-move search", 29584, True),
    ("lookahead_v1 + adaptive search (2-3 moves)", 41207, True),
    ("lookahead_v1 + adaptive search (2-4 moves)*", 51222, True), # seeds 1000-1099 (4 runs of 25 games)
]

# lookahead_v1, % of games reaching 2048 by search depth: (label, %, time per game)
DEPTHS = [
    ("network\nalone", 0, "instant"), # eval.csv at 3M
    ("1 move", 15, "~1 s"), # eval_lookahead.csv at 3M
    ("2 moves", 75, "~4 s"), # journal, model at 2M
    ("2-3 moves\n(adaptive)", 95, "~45 s"), # journal, model at 3M
    ("2-4 moves\n(adaptive)*", 98, "~3 min"), # seeds 1000-1099, model at 3M
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_axisbelow(True)


def progression():
    figure, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    style(ax)
    labels = [label for label, _, _ in PROGRESSION][::-1] # reversed: first model at the top
    scores = [score for _, score, _ in PROGRESSION][::-1]
    colors = [WITH_SEARCH if search else NETWORK_ALONE for _, _, search in PROGRESSION][::-1]
    bars = ax.barh(labels, scores, color=colors, height=0.7)
    for bar, score in zip(bars, scores):
        ax.annotate(f"{score:,}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, max(scores) * 1.1)
    ax.set_xlabel("Mean score (100 games)", fontsize=9, color=MUTED)
    ax.set_title("From the first model to the final agent", loc="left", fontsize=11, color=INK, pad=24)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NETWORK_ALONE, WITH_SEARCH)]
    ax.legend(handles, ["Network alone", "Network + search"], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, frameon=False, fontsize=9)
    figure.text(0.01, 0.01, "Seeds 0-99, except * on fresh seeds 1000-1099.", fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig("figures/2_progression.png", dpi=150, facecolor=SURFACE)


def depths():
    figure, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    style(ax)
    labels = [f"{label}\n{time}" for label, _, time in DEPTHS]
    values = [value for _, value, _ in DEPTHS]
    colors = [NETWORK_ALONE] + [WITH_SEARCH] * (len(values) - 1) # same colors as the progression chart
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.tick_params(axis="x", colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Games reaching 2048 (%), by how far ahead the search looks", loc="left", fontsize=11, color=INK, pad=12)
    ax.set_xlabel("Search depth, and time per game on CPU", fontsize=9, color=MUTED)
    figure.text(0.01, 0.01, "lookahead_v1. Seeds 0-99, except * on fresh seeds 1000-1099. "
                "\"2 moves\" was measured after 2M training steps, the others after 3M.", fontsize=7.5, color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig("figures/3_search_depth.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    progression()
    depths()
    print("Saved figures/2_progression.png and figures/3_search_depth.png")
