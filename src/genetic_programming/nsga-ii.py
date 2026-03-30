import sys
from pathlib import Path
import time

import numpy as np
from construct_ast import create_ast, evaluate_ast

# Ensure `import src.*` works regardless of the current working directory.
# When launching `python src/genetic_programming/nsga-ii.py`, Python puts
# `src/genetic_programming` on `sys.path`, but not necessarily the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.simple_data_generation import shift_time_series, shrunk_time_series

dataset_name = sys.argv[1]
dataset_path = f"data/datasets/{dataset_name}.npz"
shift_distortion_path = f"data/distortions/shift/{dataset_name}.npy"
shrink_distortion_path = f"data/distortions/shrink/{dataset_name}.npy"

dataset = np.load(dataset_path)
shift_distortion = np.load(shift_distortion_path)
shrink_distortion = np.load(shrink_distortion_path)

series = dataset["X"]
series_shifted_all = []
series_shrunk_all = []

for i in range(series.shape[0]):
    for j in range(shift_distortion.shape[0]):
        shifted_series = shift_time_series(series[i], shift_distortion[j])
        shrunk_series = shrunk_time_series(shifted_series, 1 + shrink_distortion[j] * 0.5)
        series_shifted_all.append(shifted_series)
        series_shrunk_all.append(shrunk_series)


INITIAL_POP_SIZE = 100
MAX_TREE_DEPTH = 5
TARGET_SERIES_LENGTH = 64
RNG_SEED = 42

rng = np.random.default_rng(RNG_SEED)
population = [
    create_ast(max_depth=MAX_TREE_DEPTH, target_length=TARGET_SERIES_LENGTH, rng=rng)
    for _ in range(INITIAL_POP_SIZE)
]

start_time = time.time()

for program in population:
    for series in series_shifted_all:
        evaluated_series = evaluate_ast(program, series)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")