import numpy as np
from scipy.stats import moment


def compute_1d_moment_vector(a: np.ndarray, n_moments: int = 5) -> np.ndarray:
    std = moment(a, moment=2) ** 0.5
    if std < 1e-6:
        return np.zeros(n_moments)
    moments = [moment(a, moment=i) / std ** (i) for i in range(3, n_moments + 3)]
    return np.array(moments)
