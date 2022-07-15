import sys, os, re, json
import datasets
from datasets import load_dataset, load_metric
import random
random.seed(123)
from tqdm import trange
import numpy as np

def get_data(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    wiki_data = load_dataset("wikipedia", "20220301.en", cache_dir="/fsx/ganayu/cache")
    books_data = load_dataset("bookcorpus", cache_dir="/fsx/ganayu/cache")
    w_tr = open(data_dir + "/wikibooks_train.json", "w")
    w_val = open(data_dir + "/wikibooks_val.json", "w")
    w_test = open(data_dir + "/wikibooks_test.json", "w")
    for i in trange(len(wiki_data['train'])):
        text = wiki_data['train'][i]['text']
        # text = re.sub(r'\n(?=\n)', "", text)
        # text = text.strip()
        # text = text + "\n\n"
        # rand_num = random.randint(0,10)
        rand_num = random.random()
        content = {"text": text}
        json_str = json.dumps(content) + "\n"
        if rand_num <= 0.9:
            w_tr.write(json_str)
        elif rand_num <= 0.95:
            w_val.write(json_str)
        else:
            w_test.write(json_str)
        #if i > 10000:
        #    break
    for i in trange(len(books_data['train'])):
        text = books_data['train'][i]['text']
        # text = re.sub(r'\n(?=\n)', "", text)
        # text = text.strip()
        # text = text + "\n\n"
        # rand_num = random.randint(0,10)
        rand_num = random.random()
        content = {"text": text}
        json_str = json.dumps(content) + "\n"
        if rand_num <= 0.9:
            w_tr.write(json_str)
        elif rand_num <= 0.95:
            w_val.write(json_str)
        else:
            w_test.write(json_str)
        #if i > 10000:
        #    break
    w_tr.close()
    w_val.close()
    w_test.close()

data_dir = "/fsx/ganayu/data/bert_pretraining_data/json_format"
# get_data(data_dir)

def convert_to_datasets_format(c4_dir, dest_dir):
    train_files = [ os.path.join(c4_dir, file)  for file in os.listdir(c4_dir) if file.endswith("train.json") ]
    # train_files = [ os.path.join(c4_dir, file)  for file in os.listdir(c4_dir) if file.endswith("train_small.json") ]
    # val_files = [ os.path.join(c4_dir, file) for file in os.listdir(c4_dir) if file.endswith("val.json") ]
    val_files = [ os.path.join(c4_dir, file) for file in os.listdir(c4_dir) if file.endswith("val_small.json") ]
    # train_files = val_files
    # val_files = [ os.path.join(c4_dir, file) for file in os.listdir(c4_dir) if file.endswith("val_very_small.json") ]
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    raw_datasets = load_dataset("json", data_files={"train": train_files, "validation": val_files}, cache_dir="/fsx/ganayu/experiments/supershaper/cache", chunksize=40<<20)
    raw_datasets.save_to_disk(dest_dir)
    print(len(raw_datasets["train"]))
    print(len(raw_datasets["validation"]))
# dest_dir = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy"
dest_dir = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final"
convert_to_datasets_format(data_dir, dest_dir)
# 72M train docs
# 4M valid docs

def load_from_disk(dest_dir):
    raw_datasets = datasets.load_from_disk(dest_dir)
    print(len(raw_datasets["train"]))
    print(len(raw_datasets["validation"]))
# load_from_disk(dest_dir)

def create_random_small_valid_dataset(srcf, destf):
    import random
    random.seed(123)
    lines = []
    for line in open(srcf):
        lines.append(line.strip())
    random.shuffle(lines)
    lines = lines[0:50000]
    w = open(destf, 'w')
    for line in lines:
        w.write(line + "\n")
    w.close()
# create_random_small_valid_dataset("/fsx/ganayu/data/bert_pretraining_data/json_format/wikibooks_val.json", "/fsx/ganayu/data/bert_pretraining_data/json_format/wikibooks_val_small.json")

def dataset_stats(data_dir):
    def stats(f, datasize=-1):
        seqlens = []
        pbar = tqdm(total=datasize)
        for line in open(f):
            content = json.loads(line.strip())
            seqlens.append(len(content["text"].split()))
            pbar.update(1)
        pbar.close()
        print("mean=%.2f, max=%.2f, min=%.2f, std=%.2f"%(np.mean(seqlens), np.max(seqlens), np.min(seqlens), np.std(seqlens)))
    stats(data_dir + "/wikibooks_train.json", datasize=72419478)
    stats(data_dir + "/wikibooks_val.json", datasize=4020695)
    # mean=50.79, max=68741.00, min=1.00, std=302.25
    # mean=50.70, max=45377.00, min=1.00, std=303.11
# dataset_stats(data_dir)

def tokenization_stats(data_dir, max_seq_length=512):
    from transformers import (AutoTokenizer, DataCollatorForLanguageModeling)
    from torch.utils.data.dataloader import DataLoader
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    raw_datasets = datasets.load_from_disk(data_dir)

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [
            line
            for line in examples["text"]
            if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
    
    def tokenize_function_2(examples):
        return tokenizer(
            examples["text"], return_special_tokens_mask=True
        )

    tokenized_datasets = raw_datasets.map(tokenize_function_2, batched=True, num_proc=1, remove_columns=["text"]) #, load_from_cache_file=not False)
    

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        print('inside group texts')
        print(list(examples.keys()))
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        print(total_length)
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        for k, t in result.items():
            print(k, len(result[k]))
        print('end')
        return result
    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=1) #, load_from_cache_file=not False)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    print(len(train_dataset))
    print(len(eval_dataset))
    
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=256,
        num_workers=1,
        pin_memory=True,
    )
    for step, batch in enumerate(eval_dataloader):
        print(step, batch["input_ids"].size())

# tokenization_stats(dest_dir)

