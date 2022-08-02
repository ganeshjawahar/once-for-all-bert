
'''
import sys, os, json, gzip, glob
from tqdm import tqdm


def clean_dump(src_dir, dest_dir):
    files = []
    for src_f in glob.glob(src_dir + "/*"):
        fname = src_f.split("/")[-1]
        if fname.startswith("c4-train"):
            if fname.endswith(".json.gz"):
                files.append(src_f)
    for src_f in glob.glob(src_dir + "/*"):
        fname = src_f.split("/")[-1]
        if fname.startswith("c4-validation"):
            if fname.endswith(".json.gz"):
                files.append(src_f)
    files = [files[0]] + [files[-1]]
    pbar = tqdm(total=len(files))
    num_errors = 0
    for src_f in files:
        fname = src_f.split("/")[-1]
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # w_f = gzip.open(dest_dir + "/" + fname.replace("json.gz", "json"), "wb")
        w_f = open(dest_dir + "/" + fname.replace("json.gz", "json"), "w")
        for line in gzip.open(src_f, 'rb'):
            try:
                content = json.loads(line.strip())
                #w_f.write(line.strip())
                w_f.write(json.dumps(content))
                w_f.write("\n")
            except:
                num_errors += 1
                continue
        w_f.close()
        pbar.update(1)
    pbar.close()
    print(num_errors)

clean_dump("/fsx/ganayu/data/supershaper_pretraining_data/c4/en", "/fsx/ganayu/data/supershaper_pretraining_data/c4/en_cleaned_dummy")
'''

import os
import datasets
from datasets import load_dataset, load_metric

c4_dir = "/fsx/ganayu/data/supershaper_pretraining_data/c4/en_cleaned_dummy"
train_files = [ os.path.join(c4_dir, file)  for file in os.listdir(c4_dir) if file.startswith("c4-train") ]
val_files = [ os.path.join(c4_dir, file) for file in os.listdir(c4_dir) if file.startswith("c4-validation") ]
train_files = sorted(train_files)
val_files = sorted(val_files)
raw_datasets = load_dataset("json", data_files={"train": train_files, "validation": val_files}, cache_dir="/fsx/ganayu/experiments/supershaper/cache")
raw_datasets.save_to_disk("/fsx/ganayu/data/supershaper_pretraining_data/c4_datasets_dummy")
print(len(raw_datasets["train"]))
print(len(raw_datasets["validation"]))
