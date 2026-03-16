import numpy as np
import os

n_shifts = 5
n_shrunks = 5

np.random.seed(42)

os.makedirs("data/distortions", exist_ok=True)
for distortion_name in ["shift", "shrink"]:
    os.makedirs(os.path.join("data/distortions", distortion_name), exist_ok=True)


for dataset_name in os.listdir("data/datasets"):
    data_path = os.path.join("data/datasets", dataset_name)
    data = np.load(data_path)
    data = data["X"]
    print(data.shape)

    random_shifts = np.random.random(n_shifts) / 8 + (1 / 8)
    random_shifts = random_shifts * np.power(-1, np.random.randint(0, 1 + 1, n_shifts))
    random_shifts = random_shifts * data.shape[1]
    random_shifts = random_shifts.astype(int)

    save_path = os.path.join("data/distortions/shift", dataset_name[:-4] + ".npy")
    np.save(save_path, random_shifts)

    random_shrunks = np.random.random(n_shrunks) / 4

    save_path = os.path.join("data/distortions/shrink", dataset_name[:-4] + ".npy")
    np.save(save_path, random_shrunks)
