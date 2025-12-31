import numpy as np


def simple_statistics_vector(X: np.ndarray) -> np.ndarray:
    v = []

    v.append(np.mean(X))
    v.append(np.median(X))

    v.append(np.std(X))
    v.append(np.var(X))
    v.append(np.max(X) - np.min(X))

    v.append(np.min(X))
    v.append(np.max(X))

    q25, q75 = np.percentile(X, [25, 75])
    v.append(q75 - q25)

    v.append(np.sqrt(np.mean(X**2)))

    return np.array(v)
