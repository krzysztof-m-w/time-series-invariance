import sys
from pathlib import Path
import time

import numpy as np
from construct_ast import compile_ast, create_ast

# Ensure `import src.*` works regardless of the current working directory.
# When launching `python src/genetic_programming/nsga-ii.py`, Python puts
# `src/genetic_programming` on `sys.path`, but not necessarily the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.evaluations.helpers import fdr_score
from src.data_generation.simple_data_generation import shift_time_series, shrunk_time_series

def nsga_ii_selection(programs, scores, n_select):
    """
    Select n_select individuals based on scores using NSGA-II selection.

    `scores` is always a 2D numpy array of shape (num_individuals, num_objectives).
    """
    scores_arr = np.asarray(scores)
    population_size = len(programs)

    # --- Non-dominated Sorting ---
    def dominates(a, b):
        return np.all(a >= b) and np.any(a > b)

    domination_counts = np.zeros(population_size, dtype=int)
    dominated_sets = [[] for _ in range(population_size)]
    ranks = np.full(population_size, -1, dtype=int)
    fronts = []

    for p in range(population_size):
        for q in range(population_size):
            if dominates(scores_arr[p], scores_arr[q]):
                dominated_sets[p].append(q)
            elif dominates(scores_arr[q], scores_arr[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            ranks[p] = 0
    current_front = np.where(ranks == 0)[0].tolist()
    i = 0
    while current_front:
        fronts.append(current_front)
        next_front = []
        for p in current_front:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0 and ranks[q] == -1:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        current_front = next_front

    # --- Crowding Distance Calculation ---
    def compute_crowding_dist(front):
        dists = np.zeros(len(front))
        m = scores_arr.shape[1]
        for j in range(m):
            sorted_idx = np.argsort(scores_arr[front, j])
            vals = scores_arr[front, j][sorted_idx]
            dists[sorted_idx[0]] = dists[sorted_idx[-1]] = np.inf
            if vals[-1] != vals[0]:
                dists[sorted_idx[1:-1]] += (
                    (vals[2:] - vals[:-2]) / (vals[-1] - vals[0])
                )
        return dists

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected += front
        else:
            crowding = compute_crowding_dist(front)
            sorted_crowded = [front[i] for i in np.argsort(-crowding)]
            n_remaining = n_select - len(selected)
            selected += sorted_crowded[:n_remaining]
            break

    sort_indices = selected[:n_select]
    selected_programs = [programs[i] for i in sort_indices]
    selected_scores = [scores[i] for i in sort_indices]
    return selected_programs, selected_scores

dataset_name = sys.argv[1]
dataset_path = f"data/datasets/{dataset_name}.npz"
shift_distortion_path = f"data/distortions/shift/{dataset_name}.npy"
shrink_distortion_path = f"data/distortions/shrink/{dataset_name}.npy"

dataset = np.load(dataset_path)
shift_distortion = np.load(shift_distortion_path)
shrink_distortion = np.load(shrink_distortion_path)

series_all = dataset["X"]
series_all = series_all[:10]
series_shifted_all = []
series_shrunk_all = []

for i in range(series_all.shape[0]):
    for j in range(shift_distortion.shape[0]):
        shifted_series = shift_time_series(series_all[i], shift_distortion[j])
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

# Pre-compile each AST once, so evaluation over many series avoids recursive
# traversal and per-node primitive lookups.
compiled_population = [compile_ast(program) for program in population]

scores = []

start_time = time.time()

for evaluate_program in compiled_population:
    series_embeddings = []
    series_embeddings_shrunk = []
    series_embeddings_shifted = []
    for series in series_all:
        series_embedding = evaluate_program(series)
        series_embeddings.append(series_embedding)
    series_embeddings = np.array(series_embeddings)
    for series in series_shifted_all:
        series_embedding_shifted = evaluate_program(series)
        series_embeddings_shifted.append(series_embedding_shifted)
    series_embeddings_shifted = np.array(series_embeddings_shifted)
    for series in series_shrunk_all:
        series_embedding_shrunk = evaluate_program(series)
        series_embeddings_shrunk.append(series_embedding_shrunk)
    series_embeddings_shrunk = np.array(series_embeddings_shrunk)

    series_embeddings_shifted = series_embeddings_shifted.reshape(
        series_embeddings.shape[0], -1, series_embeddings.shape[1])

    series_embedding_shifted = np.concatenate(
        [series_embeddings_shifted, series_embeddings[:, None, :]], axis=1
        )


    series_embeddings_shrunk = series_embeddings_shrunk.reshape(
        series_embeddings.shape[0], -1, series_embeddings.shape[1])

    series_embedding_shrunk = np.concatenate(
        [series_embeddings_shrunk, series_embeddings[:, None, :]], axis=1
        )

    fdr_score_shifted = fdr_score(series_embedding_shifted)
    fdr_score_shrunk = fdr_score(series_embedding_shrunk)

    fdr_score_shifted = fdr_score_shifted if not np.isnan(fdr_score_shifted) else 0
    fdr_score_shrunk = fdr_score_shrunk if not np.isnan(fdr_score_shrunk) else 0

    # Two-objective maximization: invariance to shift and shrink separately.
    scores.append([fdr_score_shifted, fdr_score_shrunk])

# Example usage in your pipeline:
N_SELECT = 10  # or a smaller number if you plan to shrink

next_population, next_scores = nsga_ii_selection(population, scores, N_SELECT)
population = next_population
scores = next_scores

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(f"Selected {len(population)} programs")
print(f"Selected scores: {scores}")