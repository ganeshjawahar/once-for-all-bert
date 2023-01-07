'''
our evolutionary search
+ HAT defaults
+ Supernet directly for calculating fitness
'''

import time, sys, os, datasets, random, copy, xlwt, json
import argparse
from sampling import get_supertransformer_config
import transformers
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, set_seed, AutoConfig, PretrainedConfig, BertConfig
from custom_layers import custom_bert
import torch
from torch.utils.data.dataloader import DataLoader
from datasets import load_metric, load_dataset
import math
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from tqdm.auto import tqdm
from utils.module_proxy_wrapper import ModuleProxyWrapper
from sampling import Sampler
from utils import calculate_params_from_config
import numpy as np
from xlrd import open_workbook
from torchinfo import summary

def convert_to_dict(string):
    _dict = json.loads(
        string.replace("BertConfig ", "").replace("\n", "").replace('""', '"')
    )
    return BertConfig(**_dict)

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
        self.trial_run = args.trial_run
        self.finetune = args.finetune
        self.expert_search = args.expert_search
        self.max_experts = args.max_experts
        self.expert_routing_type = args.expert_routing_type
        self.last_expert_averaging_expert = args.last_expert_averaging_expert
        self.fixed_arch = args.fixed_arch
        self.include_smallest_arch_into_initial_popu = args.include_smallest_arch_into_initial_popu
        if self.trial_run == "yes":
            # self.population_size = 10
            self.parent_size = 5
            self.mutation_size = 5
            self.crossover_size = 5
            self.evo_iter = 3
        self.enumerate_all = args.enumerate_all
            
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
        self.fitness_metric = args.fitness_metric
        
        # data
        self.data_dir = args.data_dir
        self.num_batches = -1
        if self.trial_run == "yes":
            self.num_batches = 1

        # constraints
        self.params_min, self.params_max = int(args.params_constraints.split(",")[0]), int(args.params_constraints.split(",")[1])

        # output
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # prepare fitness function
        self.prepare_fitness_fn_helper()

        # check fitness score of supernet
        global_metrics = self.fitness_score(self.global_config, track_progress=True)
        if self.accelerator.is_main_process:
            print(global_metrics)

        # set search space sampler
        self.sampler = Sampler("random", "none", args.mixing, self.global_config, search_space_id=self.search_space_id)
        self.seed_count = 0
        self.elastic_keys = self.get_elastic_keys()
        self.gene_choices, self.gene_names, self.elastickey2ranges = self.get_gene_choices()
        self.sample_keys = ["sample_hidden_size", "sample_intermediate_size", "sample_num_attention_heads", "sample_num_hidden_layers"]
    
    def prepare_fitness_fn_helper(self):
        if self.finetune:
            self.prepare_finetune_fitness_fn()
        else:
            self.prepare_pretrain_fitness_fn()
    
    def prepare_finetune_fitness_fn(self):
        # set seed
        set_seed(self.seed)

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        param = DistributedDataParallelKwargs(find_unused_parameters=True, check_reduction=False)
        self.accelerator = Accelerator(fp16=True, kwargs_handlers=[param])

        raw_datasets = load_dataset("glue", self.finetune)
        is_regression = self.finetune == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }
        sentence1_key, sentence2_key = task_to_keys[self.finetune]

        # load supernet config
        # self.global_config = get_supertransformer_config(self.bert_backbone, mixing=self.mixing, search_space_id=self.search_space_id)
        # set defaults like max_seq_length
        # self.global_config.max_seq_length = self.max_seq_length
        # self.global_config.alpha_divergence = 0
        # self.global_config.rewire = 0
        # self.global_config.layer_drop_prob = 0.0
        # self.global_config.hidden_dropout_prob = 0

        # Load pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_backbone, use_fast=True)
        self.global_config = AutoConfig.from_pretrained(self.supernet_ckpt_dir, num_labels=num_labels, finetuning_task=self.finetune)
        self.global_config.num_labels = num_labels
        self.model = custom_bert.BertForSequenceClassification.from_pretrained(self.supernet_ckpt_dir, config=self.global_config)

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
            self.model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in self.model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                print(
                    f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                    "Using it!"
                )
                label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
            else:
                print(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        
        if label_to_id is not None:
            self.model.config.label2id = label_to_id
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}
        padding = False

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*texts, padding=padding, max_length=self.max_seq_length, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
        
        eval_dataset = processed_datasets["validation_matched" if self.finetune == "mnli" else "validation"]

        # Log a few random samples from the eval set:
        for index in random.sample(range(len(eval_dataset)), 3):
            print(f"Sample {index} of the training set: {eval_dataset[index]}.")
        
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None))
        self.eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)

        self.model, self.eval_dataloader = self.accelerator.prepare(self.model, self.eval_dataloader)
        if (self.accelerator.distributed_type == DistributedType.MULTI_GPU or self.accelerator.distributed_type == DistributedType.TPU):
            # forward missing getattr and state_dict/load_state_dict to orig model
            self.model = ModuleProxyWrapper(self.model)
        
        # set metric
        self.metric = load_metric("glue", self.finetune)

        # cache for evaluations
        self.config_cache = {}

    def prepare_pretrain_fitness_fn(self):
        # set seed
        set_seed(self.seed)

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        param = DistributedDataParallelKwargs(find_unused_parameters=True, check_reduction=False)
        self.accelerator = Accelerator(fp16=True, kwargs_handlers=[param])

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_backbone, use_fast=True)

        # load data
        eval_dataset = datasets.load_from_disk(self.data_dir)["validation"]
        if "graphcore" in self.data_dir:
            eval_dataset = eval_dataset.remove_columns(["next_sentence_label", "labels"])
        self.dataset_len = len(eval_dataset)
        for index in random.sample(range(len(eval_dataset)), 3):
            print(f"Sample {index} of the training set: {eval_dataset[index]}.")

        # load supernet config
        self.global_config = get_supertransformer_config(self.bert_backbone, mixing=self.mixing, search_space_id=self.search_space_id, max_experts=args.max_experts, expert_routing_type=args.expert_routing_type, hypernet_hidden_size=self.hypernet_hidden_size)
        # set defaults like max_seq_length
        self.global_config.max_seq_length = self.max_seq_length
        self.global_config.alpha_divergence = 0
        self.global_config.rewire = 0
        self.global_config.layer_drop_prob = 0.0
        self.global_config.hidden_dropout_prob = 0

        # create model
        self.model = custom_bert.BertForMaskedLM.from_pretrained(self.supernet_ckpt_dir, config=self.global_config)

        '''
        # trial: set expert 2
        self.global_config.max_experts = 2
        self.global_config.expert_routing_type = "sentence"
        self.global_config.sample_expert_ids = [0 for i in range(12)]
        self.global_config.last_expert_averaging_expert = 0
        self.model = custom_bert.BertForMaskedLM.from_pretrained(self.supernet_ckpt_dir, config=self.global_config)
        self.use_expert_id = 1
        for layer_id in range(self.global_config.sample_num_hidden_layers):
            for main_expert, other_experts in [("bert.encoder.layer.<layer_id>.intermediate.dense.weight", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.intermediate.dense.bias", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.dense.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.output.dense.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.weight"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.bias")]:
                interested_expert = self.model.state_dict()[other_experts.replace("<layer_id>", str(layer_id)).replace("<expert_id>", str(self.use_expert_id-1))].data
                self.model.state_dict()[main_expert.replace("<layer_id>", str(layer_id))].data.copy_(interested_expert)
        '''

        # Data collator
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size, num_workers=1, pin_memory=True, shuffle=False) # set shuffle to always True
        self.model, self.eval_dataloader = self.accelerator.prepare(self.model, self.eval_dataloader)
        if (self.accelerator.distributed_type == DistributedType.MULTI_GPU or self.accelerator.distributed_type == DistributedType.TPU):
            self.model = ModuleProxyWrapper(self.model)

        # set metric
        self.metric = load_metric("custom_metrics/mlm_accuracy.py")

        # cache for evaluations
        self.config_cache = {}
    
    def fitness_score(self, bert_config, hashcode=None, track_progress=False):
        if self.finetune:
            return self.fitness_finetune_score(bert_config, hashcode=hashcode, track_progress=track_progress)
        else:
            return self.fitness_pretrain_score(bert_config, hashcode=hashcode, track_progress=track_progress)

    def fitness_finetune_score(self, bert_config, hashcode=None, track_progress=False):
        if hashcode and hashcode in self.config_cache:
            return self.config_cache[hashcode]
            
        self.model.set_sample_config(bert_config, drop_layers=False)

        self.model.eval()

        # valid acc for supernet
        for step, batch in enumerate(self.eval_dataloader):
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not self.finetune == "stsb" else outputs.logits.squeeze()
            self.metric.add_batch(
                predictions=self.accelerator.gather(predictions),
                references=self.accelerator.gather(batch["labels"]),
            )
        eval_metric  = self.metric.compute()
        if "accuracy" in eval_metric:
            eval_metric["accuracy"] = 100.0*eval_metric["accuracy"]
        if hashcode:
            self.config_cache[hashcode] = eval_metric

        return eval_metric
    
    def fitness_pretrain_score(self, bert_config, hashcode=None, track_progress=False):
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
        self.model.eval()
        timer_set = False
        if timer_set:
            timers = {}
            timers["forward"] = 0
            timers["gather_loss"] = 0
            timers["argmax"] = 0
            timers["pad_across_processes"] = 0
            timers["gather_predictions"] = 0
            timers["gather_labels"] = 0
            timers["get_labels"] = 0
            timers["rest"] = 0
            timers["cat"] = 0
            timers["metric_compute"] = 0
            timers["ppl_compute"] = 0

        for step, batch in enumerate(self.eval_dataloader):
            if timer_set:
                cur_time = time.time()
            with torch.no_grad():
                outputs = self.model(**batch)
            if timer_set:
                timers["forward"] += time.time()-cur_time
                cur_time = time.time()
            
            loss = outputs.loss
            losses.append(self.accelerator.gather(loss.repeat(self.per_device_eval_batch_size)))
            if timer_set:
                timers["gather_loss"] += time.time()-cur_time
                cur_time = time.time()

            predictions = outputs.logits.argmax(dim=-1)
            if timer_set:
                timers["argmax"] += time.time()-cur_time
                cur_time = time.time()
            labels = batch["labels"]
            if (not self.pad_to_max_length):  # necessary to pad predictions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(
                    predictions, dim=1, pad_index=-100
                )
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if timer_set:
                timers["pad_across_processes"] += time.time()-cur_time
                cur_time = time.time()
            predictions_gathered = self.accelerator.gather(predictions)
            if timer_set:
                timers["gather_predictions"] += time.time()-cur_time
                cur_time = time.time()
            labels_gathered = self.accelerator.gather(labels)
            if timer_set:
                timers["gather_labels"] += time.time()-cur_time
                cur_time = time.time()

            preds, refs = get_labels(predictions_gathered, labels_gathered)
            if timer_set:
                timers["get_labels"] += time.time()-cur_time
                cur_time = time.time()
            self.metric.add_batch(predictions=preds, references=refs)  # predictions and preferences are expected to be a nested list of labels, not label_ids
            if track_progress:
                progress_bar.update(1)
            if timer_set:
                timers["rest"] += time.time()-cur_time
                cur_time = time.time()
            if self.num_batches != -1 and step >= self.num_batches:
                break
        if track_progress:
            progress_bar.close()

        if timer_set:
            cur_time = time.time()
        losses = torch.cat(losses)
        losses = losses[:self.dataset_len]
        if timer_set:
            timers["cat"] += time.time()-cur_time
            cur_time = time.time()
        eval_metric = {} # self.metric.compute() super-time consuming 120s for 50K with batch size of 256
        if self.fitness_metric == "accuracy":
            eval_metric = self.metric.compute()
        if timer_set:
            timers["metric_compute"] += time.time()-cur_time
            cur_time = time.time()

        try:
            val_loss = torch.mean(losses)
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        if timer_set:
            timers["ppl_compute"] += time.time()-cur_time
            cur_time = time.time()

        eval_metric["val_loss"] = val_loss
        eval_metric["perplexity"] = perplexity
        if hashcode:
            self.config_cache[hashcode] = eval_metric

        return eval_metric

    # def convert_feature_to_hash(self, feature_config):
    #     return ("-".join([str(feat) for feat in feature_config]))

    def run_evo_search(self):
        if self.accelerator.is_main_process:
            start_time = time.time()
        if self.enumerate_all == "yes":
            all_models = self.sampler.sample_subtransformer(randomize=True, rand_seed=self.seed_count, pop_size=self.population_size, enumerate_all=True)["all_models"]
            popu = []
            for arch_config in all_models:
                feat_config = [arch_config.sample_hidden_size, arch_config.sample_num_attention_heads[0], arch_config.sample_intermediate_size[0]]
                if self.search_space_id == "v5.2":
                    feat_config.append(arch_config.sample_num_hidden_layers)
                if self.satisfy_constraint(arch_config):
                    popu.append({"arch_config": arch_config, "feat_config": feat_config})
            print("%d out of %d architectures meet the contraints"%(len(popu), len(all_models)))
            popu_scores = self.get_scores(popu)
            parents_popu = self.identify_parents(popu_scores)
            best_config = parents_popu[0]
            it = 0
            print("iteration %d, best gene: "%(it), best_config["feat_config"])
            print("iteration %d, best gene: "%(it), best_config["arch_config"])
            print("iteration %d, best gene: model size = %d, fitness score (%s) = %.2f"%(it, calculate_params_from_config(best_config["arch_config"], search_space_id=self.search_space_id), self.fitness_metric, best_config["metrics"][self.fitness_metric]))

            # write the best config and pareto front at each iteration
            self.write_to_workbook(it, best_config, [parents_popu])

            if self.accelerator.is_main_process:
                print("Evolutonary search time: %s seconds"%(time.time()-start_time))
                
            return
            
        popu = self.random_sample()
        all_iter_parents = []
        best_config = None
        for it in range(self.evo_iter):

            if self.accelerator.is_main_process:
                print("iteration %d starting with population of %d architectures..."%(it, len(popu)))
                print("evaluating %d fitness"%(it))

            popu_scores = self.get_scores(popu)

            if self.accelerator.is_main_process:
                print("iteration %d, lowest score=%.2f"%(it, min([gene["metrics"][self.fitness_metric] for gene in popu_scores]))) # assuming PPL or Loss, otherwise Max

            parents_popu = self.identify_parents(popu_scores)
            all_iter_parents.append(parents_popu)
            
            best_config = parents_popu[0]
            if self.accelerator.is_main_process:
                print("iteration %d, best gene: "%(it), best_config["feat_config"])
                print("iteration %d, best gene: model size = %d, fitness score (%s) = %.2f"%(it, calculate_params_from_config(best_config["arch_config"], search_space_id=self.search_space_id), self.fitness_metric, best_config["metrics"][self.fitness_metric]))

                # write the best config and pareto front at each iteration
                self.write_to_workbook(it, best_config, all_iter_parents)

            mutated_popu = self.mutate(parents_popu)
            crossovered_popu = self.crossover(parents_popu)

            popu = parents_popu + mutated_popu + crossovered_popu
            if self.accelerator.is_main_process:
                print("iteration %d done"%it)
        
        if self.accelerator.is_main_process:
            print("Evolutonary search time: %s seconds"%(time.time()-start_time))

            # note down best config
            print("Completed. best gene: ", best_config["arch_config"])
            print("Completed. best gene: ", best_config["feat_config"])
            print("Completed. best gene: model size = %d, fitness score (%s) = %.2f"%(calculate_params_from_config(best_config["arch_config"], search_space_id=self.search_space_id), self.fitness_metric, best_config["metrics"][self.fitness_metric]))

    def write_to_workbook(self, cur_iter, best_config, all_iter_parents):
        # create workbook
        wbook = xlwt.Workbook()

        # sheet 1 - "best_config" details
        cur_sheet = wbook.add_sheet("best_config")
        cur_sheet.write(0, 0, "best model")
        cur_sheet.write(1, 0, "gene")
        cur_sheet.write(1, 1, str(best_config["feat_config"]))
        cur_sheet.write(2, 0, "model size")
        cur_sheet.write(2, 1, str(calculate_params_from_config(best_config["arch_config"], search_space_id=self.search_space_id)))
        cur_sheet.write(3, 0, self.fitness_metric)
        cur_sheet.write(3, 1, str(best_config["metrics"][self.fitness_metric]))
        cur_sheet.write(4, 0, "config.json")
        cur_sheet.write(4, 1, str(best_config["arch_config"]))
        
        # other details - elastic_keys, gene_choices, gene_names, elastickey2ranges, search_space_id
        cur_sheet.write(6, 0, "search_space_id")
        cur_sheet.write(6, 1, self.search_space_id)
        cur_sheet.write(7, 0, "elastic_keys")
        cur_sheet.write(7, 1, str(self.elastic_keys))
        cur_sheet.write(8, 0, "gene_choices")
        cur_sheet.write(8, 1, str(self.gene_choices))
        cur_sheet.write(9, 0, "gene_names")
        cur_sheet.write(9, 1, str(self.gene_names))
        cur_sheet.write(10, 0, "elastickey2ranges")
        cur_sheet.write(10, 1, str(self.elastickey2ranges))

        # rest of the sheets - one sheet per iteration
        for si in range(len(all_iter_parents)):
            cur_sheet = wbook.add_sheet("iter_%d"%si)
            cur_sheet.write(0, 0, "rank")
            cur_sheet.write(0, 1, "gene")
            cur_sheet.write(0, 2, "model size")
            cur_sheet.write(0, 3, self.fitness_metric)
            cur_sheet.write(0, 4, "config.json")
            for ci, child in enumerate(all_iter_parents[si]):
                cur_sheet.write(ci+1, 0, ci)
                cur_sheet.write(ci+1, 1, str(child["feat_config"]))
                cur_sheet.write(ci+1, 2, str(calculate_params_from_config(child["arch_config"], search_space_id=self.search_space_id)))
                cur_sheet.write(ci+1, 3, str(child["metrics"][self.fitness_metric]))
                cur_sheet.write(ci+1, 4, str(child["arch_config"]))

        # save workbook
        wbook.save(self.output_dir + "/evo_results_%d.xls"%(cur_iter))
        
    def crossover(self, genes):
        if len(genes) < 2:
            return []
        crossover_genes = []
        k = 0
        while k < self.crossover_size:
            cur_genes = random.sample(genes, 2)
            crossover_gene = []
            for gi in range(len(cur_genes[0]["feat_config"])):
                if np.random.uniform() < 0.5:
                    crossover_gene.append(cur_genes[0]["feat_config"][gi])
                else:
                    crossover_gene.append(cur_genes[1]["feat_config"][gi])
            new_gene = {"feat_config": crossover_gene, "arch_config": self.feature2arch(crossover_gene)}
            if self.satisfy_constraint(new_gene["arch_config"]) and self.notduplicategene(new_gene["feat_config"], crossover_genes, another_population=genes):
                assert(len(new_gene)==2)
                assert(isinstance(new_gene["arch_config"],  transformers.models.bert.configuration_bert.BertConfig))
                assert(isinstance(new_gene["feat_config"], list))
                crossover_genes.append(new_gene)
                k += 1
            else:
                del new_gene
        return crossover_genes

    def mutate(self, genes):
        if len(genes) == 0:
            return []
        mutated_genes = []
        k = 0
        while k < self.mutation_size:
            cur_gene = random.choice(genes)
            # mutate this gene
            mutated_gene = []
            for gi in range(len(cur_gene["feat_config"])):
                if np.random.uniform() < self.mutation_prob:
                    mutated_gene.append(random.choice(self.gene_choices[gi]))
                else:
                    mutated_gene.append(cur_gene["feat_config"][gi])
            new_gene = {"feat_config": mutated_gene, "arch_config": self.feature2arch(mutated_gene)}
            if self.satisfy_constraint(new_gene["arch_config"]) and self.notduplicategene(new_gene["feat_config"], mutated_genes, another_population=genes):
                assert(len(new_gene)==2)
                assert(isinstance(new_gene["arch_config"],  transformers.models.bert.configuration_bert.BertConfig))
                assert(isinstance(new_gene["feat_config"], list))
                mutated_genes.append(new_gene)
                k += 1
            else:
                del new_gene
        return mutated_genes

    def identify_parents(self, genes):
        #temp_genes = []
        #for gene in genes:
        #    assert(len(gene.keys())==3)
        #    assert isinstance(gene["metrics"][self.fitness_metric], float), f"{gene['metrics'][self.fitness_metric]} is not float"
        #    temp_genes.append((gene["metrics"][self.fitness_metric], gene ))
        #temp_genes.sort()
        reverse = False
        if self.fitness_metric == "accuracy":
            reverse = True
        genes.sort(key=lambda x: x["metrics"][self.fitness_metric], reverse=reverse)
        return genes[0:self.parent_size]

    def get_scores(self, genes):
        scores = []
        progress_bar = tqdm(range(0, len(genes)), disable=not self.accelerator.is_local_main_process)
        for gene in genes:
            metrics = self.fitness_score(gene["arch_config"], hashcode=str(gene["feat_config"]), track_progress=False)
            gene['metrics'] = metrics
            scores.append(gene)
            assert(len(gene)==3)
            assert(isinstance(gene["metrics"], dict))
            assert(isinstance(gene["arch_config"],  transformers.models.bert.configuration_bert.BertConfig))
            assert(isinstance(gene["feat_config"], list))
            progress_bar.update(1)
        progress_bar.close()
        return scores
    
    def random_sample(self):
        print('creating initial population...')
        population = []
        is_done = False
        tries = 0
        while is_done == False:
            cand_archs = self.generate_sample_subtransformers()
            self.seed_count += 1
            for arch_config in cand_archs:
                feat_config = self.arch2feature(arch_config)
                assert(isinstance(feat_config, list))
                assert(isinstance(arch_config, transformers.models.bert.configuration_bert.BertConfig))
                if self.satisfy_constraint(arch_config) and self.notduplicategene(feat_config, population):
                    population.append({"arch_config": arch_config, "feat_config": feat_config})
            # if len(population) >= self.population_size:
            #    is_done = True
            #    population = population[0:self.population_size]
            # tries += 1
            is_done = True
        print('initial population size = %d/%d'%(len(population), self.population_size))
        return population
    
    def generate_sample_subtransformers(self):
        if self.expert_search == "no":
            random_archs = self.sampler.sample_subtransformer(randomize=True, rand_seed=self.seed_count, pop_size=self.population_size)["random_subtransformers"]
            if self.include_smallest_arch_into_initial_popu == "yes":
                print(self.sampler.sample_subtransformer(randomize=True, rand_seed=self.seed_count, pop_size=self.population_size, sample_one_arch="yes")["smallest_subtransformer"])
                random_archs += [self.sampler.sample_subtransformer(randomize=True, rand_seed=self.seed_count, pop_size=self.population_size, sample_one_arch="yes")["smallest_subtransformer"]]
            return random_archs
        return self.random_expert_croppings()
    
    def random_expert_croppings(self):
        rb = open_workbook(args.fixed_arch, formatting_info=True)
        best_config_sheet = rb.sheet_by_name("best_config")
        print("Fixed Subnet info: Model-Size=%s, Val-PPL=%s"%(best_config_sheet.cell(2, 1).value, best_config_sheet.cell(3, 1).value))
        print("Fixed Subnet info: Gene=%s"%(best_config_sheet.cell(1, 1).value))
        self.subnet_moe_config = convert_to_dict(best_config_sheet.cell(4, 1).value)
        for attr in ["max_experts", "expert_routing_type", "sample_expert_ids", "last_expert_averaging_expert"]:
            setattr(self.subnet_moe_config, attr, getattr(self.global_config, attr))
        popu = []
        for pi in range(self.population_size):
            sample_config = copy.deepcopy(self.subnet_moe_config)
            setattr(sample_config, "sample_expert_ids", [random.randint(0, self.max_experts-1) for li in range(sample_config.num_hidden_layers)])
            popu.append(sample_config)
        return popu

    def notduplicategene(self, potential_gene_feat_config, cur_population, another_population=None):
        for cand in cur_population:
            if str(potential_gene_feat_config) == str(cand["feat_config"]):
                return False
        if another_population is not None:
            for cand in another_population:
                if str(potential_gene_feat_config) == str(cand["feat_config"]):
                    return False
        return True

    def get_elastic_keys(self):
        elastic_keys = []
        if not self.search_space_id or self.search_space_id in ["v1.1"]:
            elastic_keys = ["sample_hidden_size"]
        elif self.search_space_id == "hidden_layer_elastic":
            elastic_keys = ["sample_hidden_size", "sample_num_hidden_layers"]
        elif self.search_space_id in ["ffn_intermediate_ratio_elastic", "v1.2"]:
            elastic_keys = ["sample_hidden_size", "sample_intermediate_size"] # inter_ratio
        elif self.search_space_id == "v2":
            elastic_keys = ["sample_hidden_size", "sample_intermediate_size", "sample_num_hidden_layers"]
        elif self.search_space_id in ["v3"] or self.search_space_id.startswith("v4"):
            elastic_keys = ["sample_hidden_size", "sample_intermediate_size", "sample_num_attention_heads"]
        return elastic_keys
    
    def get_gene_choices(self):
        if self.expert_search == "yes":
            gene_choices = [[eid for eid in range(0, self.max_experts)] for li in range(self.global_config.num_hidden_layers)]
            gene_names = "sample_expert_ids_0"
            elastickey2ranges = {"sample_expert_ids": [eid for eid in range(0, self.max_experts)]}
            return gene_choices, gene_names, elastickey2ranges
        gene_choices = []
        gene_names = []
        sampler_choices = self.sampler.get_choices()
        num_hidden_layers = max(sampler_choices["sample_num_hidden_layers"])
        elastickey2ranges = {}
        for key in self.elastic_keys:
            if key in ["sample_hidden_size", "sample_intermediate_size", "sample_num_attention_heads"]:
                if self.search_space_id is not None and key == "sample_hidden_size" and (self.search_space_id in ["v3"] or self.search_space_id.startswith("v4")):
                    elastickey2ranges[key] = [len(gene_names), len(gene_names)+1]
                    gene_names.append(key + "_0")
                    gene_choices.append(sampler_choices[key])
                else:
                    elastickey2ranges[key] = [len(gene_names), len(gene_names)+num_hidden_layers]
                    for layer_id in range(num_hidden_layers):
                        gene_names.append(key + "_" + str(layer_id))
                        gene_choices.append(sampler_choices[key])
            elif key == "sample_num_hidden_layers":
                elastickey2ranges[key] = [len(gene_names), len(gene_names)+1]
                gene_names.append(key + "_0")
                gene_choices.append(sampler_choices[key])
        return gene_choices, gene_names, elastickey2ranges

    def arch2feature(self, arch_config):
        if self.expert_search == "yes":
            return getattr(arch_config, "sample_expert_ids")
        feature_config = []
        for key in self.elastic_keys:
            if key in ["sample_hidden_size", "sample_intermediate_size", "sample_num_attention_heads"]:
                cur_choices = self.gene_choices[len(feature_config)]
                if self.search_space_id is not None and key == "sample_hidden_size" and (self.search_space_id in ["v3"] or self.search_space_id.startswith("v4")):
                    assert(not isinstance(getattr(arch_config, key), list))
                    feature_config.append(getattr(arch_config, key))   
                else:
                    values = getattr(arch_config, key) + [max(cur_choices)] * (getattr(arch_config, "num_hidden_layers") - len(getattr(arch_config, key)))
                    feature_config.extend(values)
            elif key == "sample_num_hidden_layers":
                feature_config.append(getattr(arch_config, key))
        return feature_config
    
    def feature2arch(self, feat_config):
        if self.expert_search == "yes":
            arch_config = copy.deepcopy(self.subnet_moe_config)
            setattr(arch_config, "sample_expert_ids", feat_config)
            return arch_config
        arch_config = copy.deepcopy(self.global_config)
        for key in self.elastic_keys:
            start_idx = self.elastickey2ranges[key][0]
            end_idx = self.elastickey2ranges[key][1]
            if key == "sample_num_hidden_layers":
                setattr(arch_config, key, feat_config[start_idx])
            elif self.search_space_id is not None and key == "sample_hidden_size" and (self.search_space_id in ["v3"] or self.search_space_id.startswith("v4")):
                setattr(arch_config, key, feat_config[start_idx])
            else:
                setattr(arch_config, key, feat_config[start_idx:end_idx])
        if "sample_num_hidden_layers" in self.elastickey2ranges:
            num_hidden_layers = feat_config[self.elastickey2ranges["sample_num_hidden_layers"][0]]
            for key in self.sample_keys:
                if key != "sample_num_hidden_layers":
                    setattr(arch_config, key, getattr(arch_config, key)[0:num_hidden_layers]) 
        if self.search_space_id == "v1.1":
            setattr(arch_config, "sample_intermediate_size", [ 4*getattr(arch_config, "sample_hidden_size")[i] for i in range(getattr(arch_config, "sample_num_hidden_layers")) ])
        return arch_config

    def satisfy_constraint(self, arch_config):
        cur_params = calculate_params_from_config(arch_config, search_space_id=self.search_space_id)
        if cur_params < self.params_min or cur_params > self.params_max:
            return False
        if self.search_space_id == "v1.2":
            for hid_size, inter_size in zip(getattr(arch_config, "sample_hidden_size"), getattr(arch_config, "sample_intermediate_size")):
                if inter_size < 2*hid_size:
                    return False
        return True

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
        type=float,
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
    parser.add_argument(
        "--trial_run",
        type=str,
        default="no",
        help="trial run for debugging",
    )
    parser.add_argument(
        "--enumerate_all",
        type=str,
        default="no",
        help="enumerate all architectures",
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
        help=f"change default search space: None (supershaper, default), attn_elastic, ffn_intermediate_elastic",
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

    parser.add_argument(
        "--fitness_metric",
        type=str,
        default="perplexity",
        help="pretraining accuracy metric: accuracy, val_loss, perplexity",
    )

    parser.add_argument(
        "--expert_search",
        type=str,
        default="no",
        help="yes for FFN cropping search for supernet",
    )

    parser.add_argument(
        "--max_experts",
        type=int,
        default=-1,
        help=f"number of experts for supernet training",
    )

    parser.add_argument(
        "--expert_routing_type",
        type=str,
        default="sentence",
        help=f"sentence (batch) or token level routing or sentence_single",
    )

    parser.add_argument(
        "--last_expert_averaging_expert",
        type=str,
        default="no",
        help=f"is last expert simply an average of rest of the experts?",
    )

    parser.add_argument(
        "--include_smallest_arch_into_initial_popu",
        type=str,
        default="no",
        help=f"should we include smallest arch. into initial population?",
    )

    parser.add_argument(
        "--fixed_arch",
        type=str,
        default=None,
        help=f"path to searched architecture. works if expert_search=yes. searches only for experts to crop.",
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
        default="/tmp/",
        help="Directory to write pareto front and best config",
    )

    # finetune vs. pretraining accuracy
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        help="None: only pretraining accuracy; mnli: supernet is finetuned on mnli"
    )

    args = parser.parse_args()
    print(args)

    return args

def search(args):
    evolution = EvoSearch(args)
    evolution.run_evo_search()


if __name__ == "__main__":
    args = parse_args()
    search(args)


