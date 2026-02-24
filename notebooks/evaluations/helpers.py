from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score

from src.data_generation.simple_data_generation import (
    shift_time_series,
    shrunk_time_series,
)


def grouped_bars(
    data, labels=None, series_labels=None, width=0.8, logarithmic_scale=True
):
    """
    data: 2D array-like shaped (n_series, n_categories)
    labels: labels for x-axis categories (length n_categories)
    series_labels: legend labels for each series (length n_series)
    width: total width of each group
    logarithmic_scale: if True, use logarithmic y-axis
    """
    data = np.asarray(data)
    n_series, n_cat = data.shape

    # Find maximal finite value
    finite_max = np.max(data[np.isfinite(data)])

    x = np.arange(n_cat)
    bar_w = width / n_series

    # Keep track of which bars were originally inf
    inf_mask = np.isinf(data)
    data_clipped = np.where(inf_mask, finite_max, data)

    for i in range(n_series):
        bars = plt.bar(
            x + (i - n_series / 2) * bar_w + bar_w / 2,
            data_clipped[i],
            width=bar_w,
            label=None if series_labels is None else series_labels[i],
            edgecolor="red",  # highlight bars with original inf
        )

        # Add hatch or mark for bars that were inf
        for j, bar in enumerate(bars):
            if inf_mask[i, j]:
                bar.set_hatch("//")  # hatch pattern
                bar.set_edgecolor("red")

    if labels is not None:
        plt.xticks(x, labels)

    if series_labels is not None:
        plt.legend()

    # Apply logarithmic scale if requested
    if logarithmic_scale:
        plt.yscale("log")

    plt.tight_layout()


def test_embeddings_quality(embeddings):
    stds_local = np.std(embeddings, axis=1)

    stds_global = np.std(embeddings, axis=(0, 1))

    data = np.vstack([stds_global, np.mean(stds_local, axis=0)])

    grouped_bars(data, series_labels=["global", "local"])

    plt.title("Local std vs global std")
    plt.show()

    X = embeddings

    # normalize along the last axis
    Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)

    # local cosine similarities  -> shape (N, K, K)
    cos_sim_local = Xn @ Xn.transpose(0, 2, 1)

    # global pairwise similarities -> shape (N, N, K, K)
    cos_sim_global = Xn[:, None] @ Xn[None, :].transpose(  # (N,1,K,D)
        0, 1, 3, 2
    )  # (1,N,D,K)
    mean_cos_sim = np.mean(cos_sim_global, axis=(2, 3))

    sns.heatmap(mean_cos_sim)
    plt.title("cosine similarity between vector groups")
    plt.show()

    glob_cos_sim = np.mean(mean_cos_sim)
    mask = np.eye(mean_cos_sim.shape[0]).astype(np.bool)
    local_cos_sim = np.mean(mean_cos_sim[mask])
    print(
        f"Global cos similarity: {glob_cos_sim}, Local cos similarity: {local_cos_sim}"
    )
    embedding_flattened = embeddings.reshape((-1, embeddings.shape[-1]))
    labels = np.repeat(np.arange(embeddings.shape[0]), embeddings.shape[1])

    sil_score_cos = silhouette_score(embedding_flattened, labels, metric="cosine")
    print(f"Silhouette score cosine: {sil_score_cos}")
    sil_score_cos = silhouette_score(embedding_flattened, labels, metric="euclidean")
    print(f"Silhouette score euclidean: {sil_score_cos}")

    mu_intra = np.mean(cos_sim_local)
    mu_inter = np.mean(cos_sim_global)
    sigma_intra = np.std(cos_sim_local, ddof=1)  # sample std
    sigma_inter = np.std(cos_sim_global, ddof=1)

    FDR = (mu_intra - mu_inter) ** 2 / (sigma_intra**2 + sigma_inter**2)
    print("Fisher Discriminant Ratio (FDR):", FDR)

    s_pooled = np.sqrt((sigma_intra**2 + sigma_inter**2) / 2)
    cohen_d = (mu_intra - mu_inter) / s_pooled
    print("Cohen's d:", cohen_d)


def apply_embedding_function_shifts(
    data: np.ndarray, embedding_function: callable, random_shifts: list
):
    all_embedding_vectors = []

    for i in range(data.shape[0]):
        embedding_vectors = [embedding_function(data[i])]

        for shift in random_shifts:
            shifted = shift_time_series(data[i], shift)
            embedding_vectors.append(embedding_function(shifted))

        all_embedding_vectors.append(embedding_vectors)
    return np.array(all_embedding_vectors)


def apply_embedding_function_shrunk(
    data: np.ndarray, embedding_function: callable, random_shrunks: list
):
    all_embedding_vectors = []

    for i in range(data.shape[0]):
        embedding_vectors = [embedding_function(data[i])]

        for shrunk in random_shrunks:
            shifted = shrunk_time_series(data[i], shrunk)
            embedding_vectors.append(embedding_function(shifted))

        all_embedding_vectors.append(embedding_vectors)
    return np.array(all_embedding_vectors)
