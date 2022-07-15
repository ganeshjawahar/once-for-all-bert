'''
utilties to clean c4 dataset

motivation:  pyarrow.lib.ArrowInvalid: JSON parse error: Missing a closing quotation mark in string. in row 1
when we directly do  load_dataset("json", data_files={"train": train_files, "validation": val_files}) on dump from datasets

'''

import sys, os, json, gzip, glob
from tqdm import tqdm


def clean_dump(src_dir, dest_dir):
    files = []
    for src_f in glob.glob(src_dir + "/*"):
        fname = src_f.split("/")[-1]
        if fname.startswith("c4-train") or fname.startswith("c4-validation"):
            if fname.endswith(".json.gz"):
                files.append(src_f)
    # files = files[0:1]
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

clean_dump("/fsx/ganayu/data/supershaper_pretraining_data/c4/en", "/fsx/ganayu/data/supershaper_pretraining_data/c4/en_cleaned_final")






