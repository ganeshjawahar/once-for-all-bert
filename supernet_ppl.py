'''
utilties to predict ppl given config
'''

import sys, os, datasets, random
from sampling import get_supertransformer_config
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, set_seed
from custom_layers import custom_bert
import torch
from torch.utils.data.dataloader import DataLoader
from datasets import load_metric
import math
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from tqdm.auto import tqdm
from utils.module_proxy_wrapper import ModuleProxyWrapper

class SuperNetPPL:
    def __init__(self, supernet_ckpt_dir=None, val_data=None, model_name_or_path="bert-base-cased", mixing="bert-bottleneck", random_layer_selection_probability=0.1, max_seq_length=128, per_device_eval_batch_size=5000, max_batch_steps=3):
        set_seed(42)

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        param = DistributedDataParallelKwargs(find_unused_parameters=True, check_reduction=False)
        accelerator = Accelerator(fp16=True, kwargs_handlers=[param])

        # load data
        raw_datasets = datasets.load_from_disk(val_data)["validation"]
        
        # load supernet config
        global_config = get_supertransformer_config(model_name_or_path, mixing=mixing, additional_random_softmaxing=False, random_layer_selection_probability=random_layer_selection_probability)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if max_seq_length > tokenizer.model_max_length:
            max_seq_length = min(max_seq_length, tokenizer.model_max_length)
        print(max_seq_length)

        # set defaults like max_seq_length
        global_config.max_seq_length = max_seq_length
        global_config.alpha_divergence = 0
        global_config.rewire = 0
        global_config.layer_drop_prob = 0.0
        global_config.hidden_dropout_prob = 0

        # create model
        model = custom_bert.BertForMaskedLM.from_pretrained("bert-base-cased", config=global_config)
        #model = custom_bert.BertForMaskedLM.from_pretrained(supernet_ckpt_dir, config=global_config)
        model.resize_token_embeddings(len(tokenizer))
        checkpoints = torch.load(os.path.join(supernet_ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(checkpoints)
        model.set_sample_config(global_config, drop_layers=False)

        # tokenize data
        padding = "max_length"
        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [
                line
                for line in examples["text"]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
            load_from_cache_file=True,
        )
        if "url" in tokenized_datasets[0].keys() and "timestamp" in tokenized_datasets[0].keys():
            tokenized_datasets = tokenized_datasets.remove_columns(["url", "timestamp"])

        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        
        eval_dataloader = DataLoader(tokenized_datasets, collate_fn=data_collator,  batch_size=per_device_eval_batch_size, num_workers=1, pin_memory=True, shuffle=False) # set shuffle to always True

        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
        if (accelerator.distributed_type == DistributedType.MULTI_GPU or accelerator.distributed_type == DistributedType.TPU):
            model = ModuleProxyWrapper(model)
        
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.accelerator = accelerator
        self.len_eval_dataset = len(tokenized_datasets)
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.pad_to_max_length = False
        self.global_config = global_config
        self.config_cache = {}

        self.batches = []
        seqlens = []
        for step, batch in enumerate(self.eval_dataloader):
            #for bi in range(batch['attention_mask'].size(0)):
                #for ci in range(batch['attention_mask'].size(1)):
                #    if batch['attention_mask'][bi][ci]==0:
                #        seqlens.append(ci)
                #        break
            self.batches.append(batch)
            if step == max_batch_steps-1:
                break
        
        #import numpy as np
        #print("mean=%.2f, max=%.2f, min=%.2f, std=%.2f"%(np.mean(seqlens), np.max(seqlens), np.min(seqlens), np.std(seqlens)))
        #print(len(seqlens))

    
    def convert_feature_to_hash(self, feature_config):
        return ("-".join([str(feat) for feat in feature_config]))

    def validate_subtransformer(self, transformer_config, feature_config):
        '''
        eval_metric = {}
        eval_metric["val_loss"] = 11.0
        eval_metric["perplexity"] = random.random()
        return eval_metric
        '''
        transformer_config.layer_drop_prob = 0.0
        hash_code = self.convert_feature_to_hash(feature_config)
        if hash_code in self.config_cache:
            return self.config_cache[hash_code]
        self.model.set_sample_config(transformer_config, drop_layers=False)
        metric = load_metric("../custom_metrics/mlm_accuracy.py")

        def get_labels(predictions, references):
            # Transform predictions and references tensos to numpy arrays
            if self.accelerator.device.type == "cpu":
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

        losses = []
        #progress_bar = tqdm(
        #    range(0, len(self.eval_dataloader)),
        #    disable=not self.accelerator.is_local_main_process,
        #)
        self.model.eval()
        for step, batch in enumerate(self.batches):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            with torch.no_grad():
                outputs = self.model(**batch)

            loss = outputs.loss
            losses.append(self.accelerator.gather(loss.repeat(self.per_device_eval_batch_size)))

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if (
                not self.pad_to_max_length
            ):  # necessary to pad predictions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(
                    predictions, dim=1, pad_index=-100
                )
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            predictions_gathered = self.accelerator.gather(predictions)
            labels_gathered = self.accelerator.gather(labels)

            preds, refs = get_labels(predictions_gathered, labels_gathered)
            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # predictions and preferences are expected to be a nested list of labels, not label_ids
            # progress_bar.update(1)

        losses = torch.cat(losses)
        # losses = losses[:self.len_eval_dataset]
        eval_metric = metric.compute()

        try:
            val_loss = torch.mean(losses)
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        eval_metric["val_loss"] = val_loss
        eval_metric["perplexity"] = perplexity
        self.config_cache[hash_code] = eval_metric

        return eval_metric
    


def test():
    obj = SuperNetPPL(supernet_ckpt_dir="/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model", val_data="/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final", per_device_eval_batch_size=400)
    print(obj.validate_subtransformer(obj.global_config))
    # {'accuracy': 0.6999551450883872, 'val_loss': tensor(1.3405, device='cuda:0'), 'perplexity': 3.820793163421793}
    print(obj.validate_subtransformer(obj.global_config))
# test()

import json
from transformers import (BertConfig)
def convert_to_dict(string):
    _dict = json.loads(
        string.replace("BertConfig ", "").replace("\n", "").replace('""', '"')
    )
    return BertConfig(**_dict)

def test2():
    # obj = SuperNetPPL(supernet_ckpt_dir="/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model", val_data="/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final", per_device_eval_batch_size=256, max_seq_length=512)


    obj = SuperNetPPL(supernet_ckpt_dir="/fsx/ganayu/experiments/supershaper/jul11_wikibooks_efficient_subnet_train_more_steps/seqlen_128/c4_realnews_bert-bottleneck_none_K=1_pretraining_seqlen_128_13-07-2022-00-00-19/best_model", val_data="/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final", per_device_eval_batch_size=256, max_seq_length=128, max_batch_steps=5)#0000)

    # /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_512seqlen_08-07-2022-06-48-14/best_model

    import pandas as pd
    df = pd.read_csv("/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv")
    df["configs"] = df["configs"].map(convert_to_dict)
    subtransformer_config = df.iloc[0]["configs"] # pick first row (outputted by evo search)
    # print(subtransformer_config)
    print(subtransformer_config)
    print(obj.validate_subtransformer(subtransformer_config, [0, 1, 2]))
test2()

def test512():
    obj = SuperNetPPL(supernet_ckpt_dir="/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_512seqlen_08-07-2022-06-48-14/best_model", val_data="/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final", per_device_eval_batch_size=256, max_seq_length=512, max_batch_steps=500)
    import pandas as pd
    df = pd.read_csv("/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv")
    df["configs"] = df["configs"].map(convert_to_dict)
    subtransformer_config = df.iloc[0]["configs"] # pick first row (outputted by evo search)
    # print(subtransformer_config)
    print(subtransformer_config)
    print(obj.validate_subtransformer(subtransformer_config, [0, 1, 2]))


# test512()