import os
import numpy as np
import torch

dataset_path = "data/datasets"
torch_dataset_path = "data/torch_datasets"

for dataset_file in os.listdir(dataset_path):
    if dataset_file.endswith(".npz"):
        data = np.load(os.path.join(dataset_path, dataset_file))
        X = data["X"]
        y = data["y"]

        # Shuffle the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        y = y.astype(int)
        y -= y.min()  # Ensure labels start from 0
        print(y)

        # Split into train/val/test (80/10/10)
        n_samples = X.shape[0]
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
        X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

        os.makedirs(torch_dataset_path, exist_ok=True)

        # Save as PyTorch tensors
        torch.save(
            {"samples": torch.from_numpy(X_train), "labels": torch.from_numpy(y_train)},
            os.path.join(torch_dataset_path, dataset_file.replace(".npz", "_train.pt")),
        )
        torch.save(
            {"samples": torch.from_numpy(X_val), "labels": torch.from_numpy(y_val)},
            os.path.join(torch_dataset_path, dataset_file.replace(".npz", "_val.pt")),
        )
        torch.save(
            {"samples": torch.from_numpy(X_test), "labels": torch.from_numpy(y_test)},
            os.path.join(torch_dataset_path, dataset_file.replace(".npz", "_test.pt")),
        )
