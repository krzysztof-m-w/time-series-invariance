import os
import shutil

import torch

data_dir = "data/torch_datasets"
destination_dir = "TS-TCC/data/"

dataset_names = []
for filename in os.listdir(data_dir):
    if filename.endswith(".pt"):
        dataset_name = filename.split("_")[0]
        dataset_names.append(dataset_name)

for dataset_name in dataset_names:
    os.makedirs(os.path.join(destination_dir, dataset_name), exist_ok=True)
    unique_classes = set()
    min_size = float("inf")
    for dataset_type in ["train", "test", "val"]:
        source_path = os.path.join(data_dir, f"{dataset_name}_{dataset_type}.pt")
        destination_path = os.path.join(
            destination_dir, dataset_name, f"{dataset_type}.pt"
        )
        shutil.copyfile(source_path, destination_path)
        data = torch.load(destination_path)
        min_size = min(min_size, len(data["labels"]))
        unique_classes.update(data["labels"].unique().tolist())
        seq_len = data["samples"].shape[1]

    with open("src/scripts/ts_tcc_config_template.py", "r") as f:
        content = f.read()

    for _ in range(3):
        seq_len += 3
        seq_len = seq_len // 2

    content = content.replace("num_classes_placeholder", str(len(unique_classes)))
    print("min_size", min_size)
    batch_size = min_size // 10
    if batch_size < 2:
        batch_size = 2
    print("setting batch size to", batch_size)
    content = content.replace("batch_size_placeholder", str(batch_size))
    content = content.replace("features_len_placeholder", str(seq_len))

    if seq_len <= 15:
        timesteps = seq_len // 2 + 1
    else:
        timesteps = 10
    content = content.replace("timesteps_placeholder", str(timesteps))

    with open(f"TS-TCC/config_files/{dataset_name}_Configs.py", "w") as f:
        f.write(content)
