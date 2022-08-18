'''
utilities to check logits
'''

import time, sys, os, datasets, random, copy, xlwt
import argparse
from sampling import get_supertransformer_config
import transformers
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, set_seed
from custom_layers import custom_bert
import torch
from torch.utils.data.dataloader import DataLoader
from datasets import load_metric
import math
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from tqdm.auto import tqdm
from utils.module_proxy_wrapper import ModuleProxyWrapper
from sampling import Sampler
from utils import calculate_params_from_config
import numpy as np
from torch.nn import functional as F
import torch.nn as nn

parser = argparse.ArgumentParser(
    description="Evolutionary Search"
)
parser.add_argument(
    "--seed",
    type=int,
    default=333,
    help="seed value",
)
parser.add_argument(
    "--trial_run",
    type=str,
    default="no",
    help="trial run for debugging",
)
parser.add_argument(
    "--mixing",
    type=str,
    required=True,
    help=f"specifies how to mix the tokens in bertlayers",
    choices=["attention", "gmlp", "fnet", "mobilebert", "bert-bottleneck"],
)

parser.add_argument(
    "--search_space_id",
    type=str,
    default=None,
    help=f"change default search space: None (supershaper, default), attn_elastic, ffn_intermediate_elastic",
)

parser.add_argument(
    "--supernet_ckpt_dir",
    type=str,
    default=None,
    help="Supernet checkpoint dir",
)

parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=512
)

parser.add_argument(
    "--max_seq_length",
    type=int,
    default=128,
    help="Maximum sequence length of the model",
)

parser.add_argument(
    "--bert_backbone",
    type=str,
    default="bert-base-uncased",
    help="bert public config name",
)

# data 
parser.add_argument(
    "--data_dir",
    type=str,
    default="/fsx/ganayu/data/bert_pretraining_data/wikibooks_graphcore_128_next_sentence_label_removed_w_splits",
    help=f"The directory path for tokenized dataset. We'll use validation set for search.",
)

# output dir
parser.add_argument(
    "--output_dir",
    type=str,
    default="/tmp/",
    help="Directory to write pareto front and best config",
)

args = parser.parse_args()
print(args)

# set seed
set_seed(args.seed)

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
param = DistributedDataParallelKwargs(find_unused_parameters=True, check_reduction=False)
accelerator = Accelerator(fp16=True, kwargs_handlers=[param])

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.bert_backbone, use_fast=True)

# load data
eval_dataset = datasets.load_from_disk(args.data_dir)["validation"]
if "graphcore" in args.data_dir:
    eval_dataset = eval_dataset.remove_columns(["next_sentence_label", "labels"])
dataset_len = len(eval_dataset)
for index in random.sample(range(len(eval_dataset)), 3):
    print(f"Sample {index} of the training set: {eval_dataset[index]}.")

# load supernet config
global_config = get_supertransformer_config(args.bert_backbone, mixing=args.mixing, search_space_id=args.search_space_id)
# set defaults like max_seq_length
global_config.max_seq_length = args.max_seq_length
global_config.alpha_divergence = 0
global_config.rewire = 0
global_config.layer_drop_prob = 0.0
global_config.hidden_dropout_prob = 0

# create model
model = custom_bert.BertForMaskedLM.from_pretrained(args.supernet_ckpt_dir, config=global_config)

# Data collator
# This one will take care of randomly masking the tokens.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, num_workers=1, pin_memory=True, shuffle=False) # set shuffle to always True
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
if (accelerator.distributed_type == DistributedType.MULTI_GPU or accelerator.distributed_type == DistributedType.TPU):
    model = ModuleProxyWrapper(model)
model.eval()

sampler = Sampler("random", "sandwich", args.mixing, global_config, search_space_id=args.search_space_id)
config_dict = sampler.sample_subtransformer(randomize=True, rand_seed=args.seed, pop_size=1)
super_config_small = config_dict["smallest_subtransformer"]
random_config = config_dict["random_subtransformers"][0]
print(super_config_small)
print(random_config)

def get_labels(predictions, references):
    # Transform predictions and references tensos to numpy arrays
    if accelerator.device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [str(p) for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [str(l) for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]

    return true_predictions, true_labels

def kl_divergence(teacher_logits, student_logits):
    student_logits = student_logits.reshape(-1, student_logits.size(2))
    model_output_log_prob = F.log_softmax(student_logits, dim=1)
    model_output_log_prob = model_output_log_prob.unsqueeze(2)
    
    real_output_soft = F.softmax(teacher_logits.reshape(-1, teacher_logits.size(2)), dim=1)
    real_output_soft = real_output_soft.unsqueeze(1)

    cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
    return cross_entropy_loss.squeeze()

def mse_loss(teacher_logits, student_logits):
    non_trainable_layernorm = nn.LayerNorm(teacher_logits.shape[1:], elementwise_affine=False)
    teacher_logits = non_trainable_layernorm(teacher_logits) 
    student_logits = non_trainable_layernorm(student_logits)
    mse = nn.MSELoss(reduction='none')(teacher_logits, student_logits)
    return mse.mean(-1).reshape(-1)

def get_topklogits(logits, k=5):
    logits = logits.reshape(-1, logits.size(2))
    logits = F.softmax(logits, dim=1)
    return logits.topk(k=k).values.reshape(-1)


progress_bar = tqdm(range(0, len(eval_dataloader)), disable=not accelerator.is_local_main_process)
summary_kl = {"big-rand": [], "big-small": []}
summary_mse = {"big-rand": [], "big-small": []}
summary_top5logits = {"big": [], "small": []}
for step, batch in enumerate(eval_dataloader):
    config2logits = {}
    for config_name, config_vals in [("big", global_config), ("rand", random_config), ("small", super_config_small)]:
        model.set_sample_config(config_vals, drop_layers=False)
        with torch.no_grad():
            outputs = model(**batch)
        config2logits[config_name] = outputs.logits.detach()
    
    #for name, logits_a, logits_b in [("big-rand", config2logits["big"], config2logits["rand"]), ("big-small", config2logits["big"], config2logits["small"])]:
        #kl_loss = kl_divergence(logits_a, logits_b)
        #mse_l = mse_loss(logits_a, logits_b)
        #summary_kl[name].extend(kl_loss.tolist())
        #summary_mse[name].extend(mse_l.tolist())

    logits = get_topklogits(config2logits["big"])
    summary_top5logits["big"].extend(logits.tolist())
    logits = get_topklogits(config2logits["small"])
    summary_top5logits["small"].extend(logits.tolist())
    del config2logits
    
    progress_bar.update(1)
progress_bar.close()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
for name, scores in [("top5logits", summary_top5logits)]: # ("kl", summary_kl), ("mse", summary_mse), ]:
    for split in scores:
        fig = plt.figure(figsize=(13,7))
        colors = ['b', "springgreen", "indigo", "olive", "firebrick", 'c', "gold", "violet", 'm', 'r', 'g', 'k', 'y']
        sns.histplot(data=scores[split], binwidth=0.25, bins=25)
        plt.xlabel("%s"%(name), fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        fig.savefig("%s/%s.png"%(args.output_dir, args.supernet_ckpt_dir.split("/")[-1]+"_"+name+"_"+split))
