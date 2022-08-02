import os
import datasets
from datasets import load_dataset, load_metric

def save_to_disk(src_dir, dest_dir):
    train_files = [ os.path.join(src_dir, file) for file in os.listdir(src_dir) if file.startswith("c4-train") ]
    val_files = [ os.path.join(src_dir, file) for file in os.listdir(src_dir) if file.startswith("c4-validation") ]
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    raw_datasets = load_dataset("json", data_files={"train": train_files, "validation": val_files}, cache_dir="/fsx/ganayu/experiments/supershaper/cache", chunksize=40<<20)
    raw_datasets.save_to_disk(dest_dir)
    print(len(raw_datasets["train"]))
    print(len(raw_datasets["validation"]))

src_dir = "/fsx/ganayu/data/supershaper_pretraining_data/c4/en_cleaned_final"
dest_dir = "/fsx/ganayu/data/supershaper_pretraining_data/c4_datasets_cleaned"
save_to_disk(src_dir, dest_dir)

def load_from_disk(dest_dir):
    raw_datasets = datasets.load_from_disk(dest_dir)
    print(len(raw_datasets["train"]))
    print(len(raw_datasets["validation"]))
load_from_disk(dest_dir)