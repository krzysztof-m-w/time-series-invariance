import os
from sktime.datasets import load_UCR_UEA_dataset

# The 20 datasets identified in Kurbalija et al. (2011)
# Note: some names have evolved slightly in the archive over time
kurbalija_subset = [
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


def download_kurbalija_sets(data_dir="./data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f"Starting download of {len(kurbalija_subset)} datasets...")

    for name in kurbalija_subset:
        try:
            print(f"Downloading {name}...", end=" ")
            # extract_path saves it locally so you don't have to re-download next time
            X_train, y_train = load_UCR_UEA_dataset(
                name=name, split="train", extract_path=data_dir, return_X_y=True
            )
            print("Done.")
        except Exception as e:
            print(
                f"Failed. Check if '{name}' name is correct in the current UCR version."
            )


if __name__ == "__main__":
    download_kurbalija_sets()
