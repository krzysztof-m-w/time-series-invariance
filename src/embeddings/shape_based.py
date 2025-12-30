import numpy as np
from scipy.stats import norm
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset


def paa(x, segments):
    """
    x : 1-D sequence
    segments : number of PAA segments
    """
    n = len(x)
    if n % segments == 0:
        return x.reshape((segments, -1)).mean(axis=1)
    else:
        paa_result = []
        for i in range(segments):
            start = int(i * n / segments)
            end = int((i + 1) * n / segments)
            paa_result.append(x[start:end].mean())
        return np.array(paa_result)


def sax(x, segments=4, alphabet_size=4):
    """
    x : 1-D sequence
    segments : number of PAA segments
    alphabet_size : number of discrete symbols
    """
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)

    paa_values = paa(x_norm, segments)

    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

    symbols = np.digitize(paa_values, breakpoints)
    return symbols


def dtw_features(X, prototypes):
    """
    X : list of 1-D sequences to encode
    prototypes : list of 1-D prototype sequences
    returns: array of shape (len(X), len(prototypes))
    """
    X = to_time_series_dataset(X)
    features = []
    for x in X:
        dist = [dtw(x.flatten(), p) for p in prototypes]
        features.append(dist)
    return np.array(features)
