'''
our evolutionary search
+ HAT defaults
+ Supernet directly for calculating fitness
'''

import time, sys, os, datasets, random
import argparse
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

class EvoSearch:
    def __init__(self, args):
        
        # evo-search hyperparams
        self.population_size = args.population_size
        self.parent_size = args.parent_size
        self.mutation_size = args.mutation_size
        self.crossover_size = args.crossover_size
        self.mutation_prob = args.mutation_prob
        self.evo_iter = args.evo_iter
        self.seed = args.seed

        # model hyperparams
        self.mixing = args.mixing
        self.search_space_id = args.search_space_id
        self.use_hypernet_w_low_rank = args.use_hypernet_w_low_rank
        self.bottleneck_rank = args.bottleneck_rank
        self.hypernet_hidden_size = args.hypernet_hidden_size
        self.supernet_ckpt_dir = args.supernet_ckpt_dir
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.max_seq_length = args.max_seq_length
        self.bert_backbone = args.bert_backbone
        self.pad_to_max_length = False

        # data
        self.data_dir = args.data_dir

        # constraints
        self.params_min, self.params_max = int(args.params_constraints.split(",")[0]), int(args.params_constraints.split(",")[1])

        # output
        self.output_dir = args.output_dir

        # prepare fitness function
        self.prepare_fitness_fn()

        # check fitness score of supernet
        print(self.fitness_score(self.global_config, track_progress=True))
    
    def prepare_fitness_fn(self):
        # set seed
        set_seed(self.seed)

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        param = DistributedDataParallelKwargs(find_unused_parameters=True, check_reduction=False)
        self.accelerator = Accelerator(fp16=True, kwargs_handlers=[param])

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_backbone, use_fast=True)

        # load data
        eval_dataset = datasets.load_from_disk(self.data_dir)["validation"]
        self.dataset_len = len(eval_dataset)
        for index in random.sample(range(len(eval_dataset)), 3):
            print(f"Sample {index} of the training set: {eval_dataset[index]}.")

        # load supernet config
        self.global_config = get_supertransformer_config(self.bert_backbone, mixing=self.mixing)
        # set defaults like max_seq_length
        self.global_config.max_seq_length = self.max_seq_length
        self.global_config.alpha_divergence = 0
        self.global_config.rewire = 0
        self.global_config.layer_drop_prob = 0.0
        self.global_config.hidden_dropout_prob = 0

        # create model
        self.model = custom_bert.BertForMaskedLM.from_pretrained(self.bert_bsupernet_ckpt_dirackbone, config=self.global_config)
        self.model.eval()
        self.model, self.eval_dataloader = self.accelerator.prepare(self.model, self.eval_dataloader)
        if (self.accelerator.distributed_type == DistributedType.MULTI_GPU or self.accelerator.distributed_type == DistributedType.TPU):
            model = ModuleProxyWrapper(model)

        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,  batch_size=self.per_device_eval_batch_size, num_workers=1, pin_memory=True, shuffle=False) # set shuffle to always True

        # set metric
        self.metric = load_metric("../custom_metrics/mlm_accuracy.py")

        # cache for evaluations
        self.config_cache = {}
    
    def fitness_score(self, bert_config, hashcode=None, track_progress=False):
        if hashcode and hashcode in self.config_cache:
            return self.config_cache[hashcode]
            
        self.model.set_sample_config(bert_config, drop_layers=False)

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
        if track_progress:
            progress_bar = tqdm(range(0, len(self.eval_dataloader)), disable=not self.accelerator.is_local_main_process)
        for step, batch in enumerate(self.eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            with torch.no_grad():
                outputs = self.model(**batch)

            loss = outputs.loss
            losses.append(self.accelerator.gather(loss.repeat(self.per_device_eval_batch_size)))

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if (not self.pad_to_max_length):  # necessary to pad predictions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(
                    predictions, dim=1, pad_index=-100
                )
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            predictions_gathered = self.accelerator.gather(predictions)
            labels_gathered = self.accelerator.gather(labels)

            preds, refs = get_labels(predictions_gathered, labels_gathered)
            self.metric.add_batch(predictions=preds, references=refs)  # predictions and preferences are expected to be a nested list of labels, not label_ids
            if track_progress:
                progress_bar.update(1)
        if track_progress:
            progress_bar.close()

        losses = torch.cat(losses)
        losses = losses[:self.dataset_len]
        eval_metric = self.metric.compute()

        try:
            val_loss = torch.mean(losses)
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        eval_metric["val_loss"] = val_loss
        eval_metric["perplexity"] = perplexity
        if hashcode:
            self.config_cache[hashcode] = eval_metric

        return eval_metric

    
    def convert_feature_to_hash(self, feature_config):
        return ("-".join([str(feat) for feat in feature_config]))









def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolutionary Search"
    )
    
    # evo-search hyperparams (HAT defaults)
    parser.add_argument(
        "--population_size",
        type=int,
        default=125,
        help="Population Size for Evo-Search",
    )
    parser.add_argument(
        "--parent_size",
        type=int,
        default=25,
        help="Parent Size",
    )
    parser.add_argument(
        "--mutation_size",
        type=int,
        default=50,
        help="Mutation Size",
    )
    parser.add_argument(
        "--crossover_size",
        type=int,
        default=50,
        help="Crossover Size",
    )
    parser.add_argument(
        "--mutation_prob",
        type=str,
        default=0.3,
        help="Mutation Probability",
    )
    parser.add_argument(
        "--evo_iter",
        type=int,
        default=30,
        help="#iterations for Evolutionary Search",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=333,
        help="seed value",
    )

    # model
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
        help=f"change default search space: attn_elastic, ffn_intermediate_elastic",
    )

    parser.add_argument(
        "--use_hypernet_w_low_rank",
        type=int,
        default=0,
        help=f"use HyperNetDynamicLinear. only useful if mixing is set to bert-bottleneck",
    )

    parser.add_argument(
        "--bottleneck_rank",
        type=int,
        default=50,
        help=f"use HyperNetDynamicLinear. only useful if use_hypernet_w_low_rank is set",
    )

    parser.add_argument(
        "--hypernet_hidden_size",
        type=int,
        default=64,
        help=f"set hidden size for hypernet. only useful if use_hypernet_w_low_rank is set",
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
        default=5000
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
        default=None,
        help=f"The directory path for tokenized dataset. We'll use validation set for search.",
    )
    
    # constraints
    parser.add_argument(
        "--params_constraints",
        type=str,
        default="60000000,67000000",
        help="Constraints on Parameters: min,max",
    )

    # output dir
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/fsx/ganayu/experiments/supershaper/jul10_search_results",
        help="Directory to write pareto front and best config",
    )

    args = parser.parse_args()

    return args

def search(args):
    evolution = EvoSearch(args)

    start_time = time.time()
    best_config = evolution.run_evo_search()
    
    print("Evolutonary search time: %s seconds"%(time.time()-start_time))
    for val in sorted(best_config):
        print("Best config's %s = "%val, best_config[val])


if __name__ == "__main__":
    args = parse_args()
    search(args)


