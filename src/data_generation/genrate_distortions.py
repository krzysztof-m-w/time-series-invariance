import numpy as np


data_path = "data/random_sequences/random_sequences.npy"
data = np.load(data_path)
print(data.shape)

n = 5
random_shifts = np.random.random(n) / 8 + (1 / 8)
random_shifts = random_shifts * np.power(-1, np.random.randint(0, 1 + 1, n))
random_shifts = random_shifts * data.shape[1]
random_shifts = random_shifts.astype(int)

np.save("notebooks/evaluations/data/random_shifts.npy", random_shifts)

n = 5
random_shrunks = np.random.random(n) / 4

np.save("notebooks/evaluations/data/random_shrunks.npy", random_shrunks)
