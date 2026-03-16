import os
import numpy as np
from aeon.datasets import load_classification

max_samples = 500

datasets = [
    "Adiac",
    "Fish",
    "SwedishLeaf",
    "OSULeaf",
    "50words",
    "FacesUCR",
    "FaceFour",
    "GunPoint",
    "Wafer",
    "Trace",
    "Lighting2",
    "Lighting7",
    "ECG200",
    "MedicalImages",
    "Beef",
    "Coffee",
    "OliveOil",
    "SyntheticControl",
    "CBF",
    "TwoPatterns",
]


def download_sets(data_dir="./data/datasets"):

    os.makedirs(data_dir, exist_ok=True)

    for name in datasets:
        try:
            print(f"Downloading {name}...")
            X_train, y_train = load_classification(name, split="train")
            X_test, y_test = load_classification(name, split="test")

            X_merged = np.concatenate([X_train, X_test], axis=0)
            X_merged = X_merged.reshape((X_merged.shape[0], X_merged.shape[2]))
            X_merged = X_merged[:max_samples]
            y_merged = np.concatenate([y_train, y_test], axis=0)
            y_merged = y_merged[:max_samples]
            np.savez(os.path.join(data_dir, f"{name}.npz"), X=X_merged, y=y_merged)
            print("Done")
        except Exception as e:
            print(f"{name} failed: {e}")


download_sets()
