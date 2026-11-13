from pathlib import Path
from typing import NamedTuple


# Where each model's files live: one folder per model, models/<name>/.
# Every script gets its paths from here, so they always agree.

MODELS_DIR = Path(__file__).parent / "models" # the "models" folder next to this file, wherever the scripts are run from


class ModelFiles(NamedTuple): # a NamedTuple is a small object whose values are accessed by name, e.g. files.model
    folder: Path
    model: Path # the model at the end of training
    best_model: Path # the best model seen during training (saved by score_callback.py)
    replay_buffer: Path # the saved experiences, to continue training later
    eval_csv: Path # the evaluation results during training (read by plot_runs.py)


def model_files(name):
    # "big_v1" -> models/big_v1/model.zip, models/big_v1/best_model.zip, etc.
    folder = MODELS_DIR / name.removesuffix(".zip") # accepts "big_v1.zip" too, out of habit
    return ModelFiles(
        folder=folder,
        model=folder / "model.zip",
        best_model=folder / "best_model.zip",
        replay_buffer=folder / "replay_buffer.pkl",
        eval_csv=folder / "eval.csv",
    )


def find_model_to_load(name_or_path):
    # For evaluate.py and watch_ai.py: accepts a model name ("big_v1" -> models/big_v1/model.zip)
    # or the path of a specific file (e.g. models/big_v1/best_model.zip, to play the best model instead of the final one).
    if Path(name_or_path).is_file():
        return Path(name_or_path)
    path = model_files(name_or_path).model
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path} (and {name_or_path} isn't a file either)")
    return path
