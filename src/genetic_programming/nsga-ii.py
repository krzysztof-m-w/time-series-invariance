import sys
from pathlib import Path
import time

import numpy as np
from construct_ast import AstNode, TsAstProgram, compile_ast, create_ast

# Ensure `import src.*` works regardless of the current working directory.
# When launching `python src/genetic_programming/nsga-ii.py`, Python puts
# `src/genetic_programming` on `sys.path`, but not necessarily the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.evaluations.helpers import fdr_score
from src.data_generation.simple_data_generation import shift_time_series, shrunk_time_series

def _non_dominated_sort(scores_arr: np.ndarray):
    """
    Non-dominated sort for maximization objectives.

    Returns (fronts, ranks) where:
    - fronts: list[list[int]] of indices
    - ranks: np.ndarray[int] rank per individual (0 is best front)
    """
    scores_arr = np.asarray(scores_arr, dtype=float)
    population_size = scores_arr.shape[0]

    def dominates(a, b):
        return np.all(a >= b) and np.any(a > b)

    domination_counts = np.zeros(population_size, dtype=int)
    dominated_sets = [[] for _ in range(population_size)]
    ranks = np.full(population_size, -1, dtype=int)
    fronts: list[list[int]] = []

    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue
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
        next_front: list[int] = []
        for p in current_front:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0 and ranks[q] == -1:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        current_front = next_front

    return fronts, ranks


def _crowding_distances(scores_arr: np.ndarray, front: list[int]) -> np.ndarray:
    """
    Crowding distance for a single front.

    Returns array aligned with `front` positions (not global indices).
    """
    scores_arr = np.asarray(scores_arr, dtype=float)
    if not front:
        return np.asarray([], dtype=float)

    dists = np.zeros(len(front), dtype=float)
    m = scores_arr.shape[1]
    front_scores = scores_arr[np.asarray(front)]

    for j in range(m):
        sorted_pos = np.argsort(front_scores[:, j])
        vals = front_scores[sorted_pos, j]
        dists[sorted_pos[0]] = dists[sorted_pos[-1]] = np.inf
        denom = vals[-1] - vals[0]
        if denom != 0 and len(front) > 2:
            dists[sorted_pos[1:-1]] += (vals[2:] - vals[:-2]) / denom

    return dists


def nsga_ii_selection(programs, scores, n_select):
    """
    Select n_select individuals based on scores using NSGA-II selection.

    `scores` is always a 2D numpy array of shape (num_individuals, num_objectives).
    """
    scores_arr = np.asarray(scores, dtype=float)
    n_select = int(n_select)
    if n_select <= 0:
        return [], np.empty((0, scores_arr.shape[1] if scores_arr.ndim == 2 else 0), dtype=float)

    fronts, _ = _non_dominated_sort(scores_arr)

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected += front
        else:
            crowding = _crowding_distances(scores_arr, front)
            sorted_crowded = [front[i] for i in np.argsort(-crowding)]
            n_remaining = n_select - len(selected)
            selected += sorted_crowded[:n_remaining]
            break

    sort_indices = selected[:n_select]
    selected_programs = [programs[i] for i in sort_indices]
    selected_scores = scores_arr[np.asarray(sort_indices)]
    return selected_programs, selected_scores

def _list_nodes_with_paths(node: AstNode, path=()):
    out = [(path, node)]
    if node.node_type == "unary" and node.child is not None:
        out.extend(_list_nodes_with_paths(node.child, path + ("child",)))
    elif node.node_type == "binary" and node.left is not None and node.right is not None:
        out.extend(_list_nodes_with_paths(node.left, path + ("left",)))
        out.extend(_list_nodes_with_paths(node.right, path + ("right",)))
    return out


def _replace_subtree(node: AstNode, path: tuple[str, ...], new_subtree: AstNode) -> AstNode:
    if not path:
        return new_subtree

    step = path[0]
    rest = path[1:]

    if node.node_type == "unary":
        if step != "child" or node.child is None:
            return node
        replaced_child = _replace_subtree(node.child, rest, new_subtree)
        return AstNode(
            node_type="unary",
            op_name=node.op_name,
            params=node.params,
            child=replaced_child,
        )

    if node.node_type == "binary":
        if step == "left" and node.left is not None:
            replaced_left = _replace_subtree(node.left, rest, new_subtree)
            return AstNode(
                node_type="binary",
                op_name=node.op_name,
                params=node.params,
                left=replaced_left,
                right=node.right,
            )
        if step == "right" and node.right is not None:
            replaced_right = _replace_subtree(node.right, rest, new_subtree)
            return AstNode(
                node_type="binary",
                op_name=node.op_name,
                params=node.params,
                left=node.left,
                right=replaced_right,
            )
        return node

    # input node: cannot traverse further; return as-is
    return node


def _subtree_crossover(
    p1: TsAstProgram, p2: TsAstProgram, rng: np.random.Generator
) -> tuple[TsAstProgram, TsAstProgram]:
    nodes1 = _list_nodes_with_paths(p1.root)
    nodes2 = _list_nodes_with_paths(p2.root)
    path1, sub1 = nodes1[int(rng.integers(0, len(nodes1)))]
    path2, sub2 = nodes2[int(rng.integers(0, len(nodes2)))]

    child1_root = _replace_subtree(p1.root, path1, sub2)
    child2_root = _replace_subtree(p2.root, path2, sub1)

    return (
        TsAstProgram(root=child1_root, target_length=p1.target_length),
        TsAstProgram(root=child2_root, target_length=p2.target_length),
    )


def _subtree_mutation(
    program: TsAstProgram,
    *,
    rng: np.random.Generator,
    max_tree_depth: int,
    target_length: int,
) -> TsAstProgram:
    nodes = _list_nodes_with_paths(program.root)
    path, _ = nodes[int(rng.integers(0, len(nodes)))]

    donor = create_ast(
        max_depth=int(rng.integers(1, max(2, max_tree_depth + 1))),
        target_length=int(target_length),
        rng=rng,
    )
    mutated_root = _replace_subtree(program.root, path, donor.root)
    return TsAstProgram(root=mutated_root, target_length=program.target_length)


def _tournament_select_index(
    ranks: np.ndarray, crowding: np.ndarray, rng: np.random.Generator, k: int = 2
) -> int:
    k = max(2, int(k))
    n = len(ranks)
    cand = rng.integers(0, n, size=k)
    best = int(cand[0])
    for idx in cand[1:]:
        idx = int(idx)
        if ranks[idx] < ranks[best]:
            best = idx
        elif ranks[idx] == ranks[best] and crowding[idx] > crowding[best]:
            best = idx
    return best


def _rank_and_crowding(scores_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores_arr, dtype=float)
    fronts, ranks = _non_dominated_sort(scores_arr)
    crowding = np.zeros(scores_arr.shape[0], dtype=float)
    for front in fronts:
        d = _crowding_distances(scores_arr, front)
        for pos, idx in enumerate(front):
            crowding[int(idx)] = float(d[pos])
    return ranks, crowding


def _evaluate_population(
    population: list[TsAstProgram],
    *,
    series_all: np.ndarray,
    series_shifted_all: list[np.ndarray],
    series_shrunk_all: list[np.ndarray],
    test_indices: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    compiled_population = [compile_ast(program) for program in population]
    scores = np.zeros((len(population), 2), dtype=float)
    scores_test = np.zeros((len(population), 2), dtype=float)

    for i, evaluate_program in enumerate(compiled_population):
        series_embeddings = []
        series_embeddings_shrunk = []
        series_embeddings_shifted = []

        for series in series_all:
            series_embeddings.append(evaluate_program(series))
        series_embeddings = np.asarray(series_embeddings, dtype=float)

        for series in series_shifted_all:
            series_embeddings_shifted.append(evaluate_program(series))
        series_embeddings_shifted = np.asarray(series_embeddings_shifted, dtype=float)

        for series in series_shrunk_all:
            series_embeddings_shrunk.append(evaluate_program(series))
        series_embeddings_shrunk = np.asarray(series_embeddings_shrunk, dtype=float)

        series_embeddings_shifted = series_embeddings_shifted.reshape(
            series_embeddings.shape[0], -1, series_embeddings.shape[1]
        )
        series_embedding_shifted = np.concatenate(
            [series_embeddings_shifted, series_embeddings[:, None, :]], axis=1
        )

        series_embeddings_shrunk = series_embeddings_shrunk.reshape(
            series_embeddings.shape[0], -1, series_embeddings.shape[1]
        )
        series_embedding_shrunk = np.concatenate(
            [series_embeddings_shrunk, series_embeddings[:, None, :]], axis=1
        )

        fdr_score_shifted = fdr_score(series_embedding_shifted[train_indices])
        fdr_score_shrunk = fdr_score(series_embedding_shrunk[train_indices])

        fdr_score_shifted_test = fdr_score(series_embedding_shifted[test_indices])
        fdr_score_shrunk_test = fdr_score(series_embedding_shrunk[test_indices])

        fdr_score_shifted = 0.0 if np.isnan(fdr_score_shifted) else float(fdr_score_shifted)
        fdr_score_shrunk = 0.0 if np.isnan(fdr_score_shrunk) else float(fdr_score_shrunk)

        fdr_score_shifted_test = 0.0 if np.isnan(fdr_score_shifted_test) else float(fdr_score_shifted_test)
        fdr_score_shrunk_test = 0.0 if np.isnan(fdr_score_shrunk_test) else float(fdr_score_shrunk_test)

        # Optimize on train split; report generalization on test split.
        scores[i, 0] = fdr_score_shifted
        scores[i, 1] = fdr_score_shrunk
        scores_test[i, 0] = fdr_score_shifted_test
        scores_test[i, 1] = fdr_score_shrunk_test

    return scores, scores_test


def _make_offspring(
    population: list[TsAstProgram],
    scores_arr: np.ndarray,
    *,
    rng: np.random.Generator,
    pop_size: int,
    max_tree_depth: int,
    target_length: int,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    tournament_k: int = 2,
) -> list[TsAstProgram]:
    pop_size = int(pop_size)
    ranks, crowding = _rank_and_crowding(scores_arr)

    offspring: list[TsAstProgram] = []
    while len(offspring) < pop_size:
        i1 = _tournament_select_index(ranks, crowding, rng, k=tournament_k)
        i2 = _tournament_select_index(ranks, crowding, rng, k=tournament_k)
        p1 = population[i1]
        p2 = population[i2]

        if rng.random() < float(crossover_rate):
            c1, c2 = _subtree_crossover(p1, p2, rng)
        else:
            c1, c2 = p1, p2

        if rng.random() < float(mutation_rate):
            c1 = _subtree_mutation(
                c1, rng=rng, max_tree_depth=max_tree_depth, target_length=target_length
            )
        if rng.random() < float(mutation_rate):
            c2 = _subtree_mutation(
                c2, rng=rng, max_tree_depth=max_tree_depth, target_length=target_length
            )

        offspring.append(c1)
        if len(offspring) < pop_size:
            offspring.append(c2)

    return offspring


def main() -> None:
    dataset_name = sys.argv[1]
    dataset_path = f"data/datasets/{dataset_name}.npz"
    shift_distortion_path = f"data/distortions/shift/{dataset_name}.npy"
    shrink_distortion_path = f"data/distortions/shrink/{dataset_name}.npy"

    dataset = np.load(dataset_path)
    shift_distortion = np.load(shift_distortion_path)
    shrink_distortion = np.load(shrink_distortion_path)

    series_all = dataset["X"]
    series_all = series_all[:100]
    series_shifted_all: list[np.ndarray] = []
    series_shrunk_all: list[np.ndarray] = []

    for i in range(series_all.shape[0]):
        for j in range(shift_distortion.shape[0]):
            shifted_series = shift_time_series(series_all[i], shift_distortion[j])
            shrunk_series = shrunk_time_series(shifted_series, 1 + shrink_distortion[j] * 0.5)
            series_shifted_all.append(shifted_series)
            series_shrunk_all.append(shrunk_series)

    POP_SIZE = 100
    GENERATIONS = 10
    MAX_TREE_DEPTH = 5
    TARGET_SERIES_LENGTH = 64
    RNG_SEED = 42
    CROSSOVER_RATE = 0.9
    MUTATION_RATE = 0.2
    TOURNAMENT_K = 2
    TEST_SPLIT = 0.2

    rng = np.random.default_rng(RNG_SEED)
    test_indices = rng.choice(len(series_all), size=int(len(series_all) * TEST_SPLIT), replace=False)
    train_indices = np.setdiff1d(np.arange(len(series_all)), test_indices)
    population = [
        create_ast(max_depth=MAX_TREE_DEPTH, target_length=TARGET_SERIES_LENGTH, rng=rng)
        for _ in range(POP_SIZE)
    ]

    scores, scores_test = _evaluate_population(
        population,
        series_all=series_all,
        series_shifted_all=series_shifted_all,
        series_shrunk_all=series_shrunk_all,
        test_indices=test_indices,
        train_indices=train_indices,
    )

    for gen in range(GENERATIONS):
        start_time = time.time()
        offspring = _make_offspring(
            population,
            scores,
            rng=rng,
            pop_size=POP_SIZE,
            max_tree_depth=MAX_TREE_DEPTH,
            target_length=TARGET_SERIES_LENGTH,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            tournament_k=TOURNAMENT_K,
        )

        offspring_scores, _offspring_scores_test = _evaluate_population(
            offspring,
            series_all=series_all,
            series_shifted_all=series_shifted_all,
            series_shrunk_all=series_shrunk_all,
            test_indices=test_indices,
            train_indices=train_indices,
        )

        combined_population = population + offspring
        combined_scores = np.vstack([scores, offspring_scores])
        population, scores = nsga_ii_selection(combined_population, combined_scores, POP_SIZE)

        scores, scores_test = _evaluate_population(
            population,
            series_all=series_all,
            series_shifted_all=series_shifted_all,
            series_shrunk_all=series_shrunk_all,
            test_indices=test_indices,
            train_indices=train_indices,
        )

        best = np.max(scores, axis=0)
        best_test = np.max(scores_test, axis=0)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"gen={gen+1}/{GENERATIONS} best_shift={best[0]:.4f} best_shrink={best[1]:.4f}")
        print(f"best_shift_test={best_test[0]:.4f} best_shrink_test={best_test[1]:.4f}")

    print(f"Final population size: {len(population)}")
    print(f"Final scores shape: {scores.shape}")
    print(f"Final scores_test shape: {scores_test.shape}")


if __name__ == "__main__":
    main()