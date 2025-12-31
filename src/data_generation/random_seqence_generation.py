import numpy as np


def signal_trend(n):
    t = np.linspace(0, 1, n)
    slope = np.random.uniform(-3, 3)
    intercept = np.random.uniform(-1, 1)
    noise = 0.2 * np.random.normal(0, 1, n)
    return slope * t + intercept + noise


def signal_sine(n):
    t = np.linspace(0, 1, n)
    amp = np.random.uniform(0.5, 3.0)
    freq = np.random.uniform(1, 6)
    phase = np.random.uniform(0, 2 * np.pi)
    noise = 0.15 * np.random.normal(0, 1, n)
    return amp * np.sin(2 * np.pi * freq * t + phase) + noise


def signal_piecewise(n):
    x = np.random.normal(0, 0.2, n)
    k1, k2 = np.sort(np.random.choice(range(50, n - 50), 2, replace=False))
    x[:k1] += 0.5
    x[k1:k2] += 2.0
    x[k2:] -= 1.0
    return x


def signal_ar1(n):
    phi = np.random.uniform(0.5, 0.95)
    drift = np.random.uniform(-0.05, 0.05)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + drift + np.random.normal(0, 0.2)
    return x


rng = np.random.default_rng()
PATTERNS = [signal_trend, signal_sine, signal_piecewise, signal_ar1]


def make_dataset(n_signals=50, length=300, add_scale_shift=False):
    signals = []
    for _ in range(n_signals):
        base = rng.choice(PATTERNS)(length)

        # normalize energy so families are comparable
        base = (base - base.mean()) / (base.std() + 1e-6)

        if add_scale_shift:
            a = rng.uniform(0.5, 3.0)  # scale
            b = rng.uniform(-2.0, 2.0)  # shift
            base = a * base + b

        signals.append(base)
    return np.array(signals)


if __name__ == "__main__":
    import os

    X = make_dataset(n_signals=100, length=400, add_scale_shift=True)

    os.makedirs("data/random_sequences", exist_ok=True)
    np.save("data/random_sequences/random_sequences.npy", X)
