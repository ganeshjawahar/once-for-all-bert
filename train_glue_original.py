# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os, sys
import random
import json
from copy import deepcopy

from xlrd import open_workbook
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from utils import (calculate_params_from_config, millify)
import pandas as pd

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    BertConfig
)
from transformers.utils.versions import require_version
from custom_layers import custom_bert
import torch

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

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

def convert_to_dict(string):
    _dict = json.loads(
        string.replace("BertConfig ", "").replace("\n", "").replace('""', '"')
    )
    return BertConfig(**_dict)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="Tokenizer name."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    parser.add_argument(
        "--skip_saving_checkpoints",
        type=str,
        default="no",
        help="Skip saving checkpoint whatsover",
    )
    parser.add_argument(
        "--subtransformer_config_path",
        type=str,
        default=None,
        help=f"The path to a subtransformer configration",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Warump ratio in the lr scheduler."
    )
    parser.add_argument(
        "--is_mnli_checkpoint",
        type=int,
        default=0,
        help=f"if model path is a pretrained mnli checkpoint",
    )
    parser.add_argument(
        "--all_experts_average",
        type=int,
        default=0,
        help=f"useful only when the model is MoE. performs a simple parameter average of all experts",
    )

    parser.add_argument(
        "--last_expert_averaging_expert",
        type=str,
        default="no",
        help=f"is last expert simply an average of rest of the experts?",
    )

    parser.add_argument(
        "--use_expert_id",
        type=int,
        default=0,
        help=f"set expert id to use",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    if not args.subtransformer_config_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config, ignore_mismatched_sizes=args.is_mnli_checkpoint
        )
        if hasattr(config, "torch_dtype"):
            setattr(config, "torch_dtype", None)
    else:
        '''
        if args.subtransformer_config_path.endswith(".csv"):
            df = pd.read_csv(args.subtransformer_config_path)
            df["configs"] = df["configs"].map(convert_to_dict)
            subnet_config = df.iloc[0]["configs"] # pick first row (outputted by evo search)
            subnet_config = subnet_config.to_dict()
            subnet_config["num_labels"] = num_labels
        else:
        '''
        rb = open_workbook(args.subtransformer_config_path, formatting_info=True)
        best_config_sheet = rb.sheet_by_name("best_config")
        print("Subnet info: Model-Size=%s, Val-PPL=%s"%(best_config_sheet.cell(2, 1).value, best_config_sheet.cell(3, 1).value))
        print("Subnet info: Gene=%s"%(best_config_sheet.cell(1, 1).value))
        subnet_config = convert_to_dict(best_config_sheet.cell(4, 1).value)
        elastic_keys = eval(best_config_sheet.cell(7, 1).value)
        gene_choices = eval(best_config_sheet.cell(8, 1).value)
        gene_names = eval(best_config_sheet.cell(9, 1).value) if not isinstance(best_config_sheet.cell(9, 1).value, str) else best_config_sheet.cell(9, 1).value
        elastickey2ranges = eval(best_config_sheet.cell(10, 1).value)
        print("Subnet info: Search_space_id=%s"%(best_config_sheet.cell(6, 1).value))
        print("Subnet info: elastic_keys=", elastic_keys)
        print("Subnet info: gene_choices=", gene_choices)
        print("Subnet info: gene_names=", gene_names)
        print("Subnet info: elastickey2ranges=", elastickey2ranges)
        subnet_config.num_labels = num_labels

        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        if hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_jack_1L", "archrouting_jack_2L", "archrouting_1L", "archrouting_2L"]:
            for attr in ["max_experts", "expert_routing_type", "last_expert_averaging_expert", "sample_expert_ids", "fixed_hypernet_input", "hypernet_hidden_size"]:
                if hasattr(config, attr):
                    setattr(subnet_config, attr, getattr(config, attr))

        # subnet_config.hidden_dropout_prob = 0.1
        model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=subnet_config, ignore_mismatched_sizes=args.is_mnli_checkpoint)
        # checkpoints = torch.load(
        #    os.path.join(args.model_name_or_path, "pytorch_model.bin"),
        #    map_location="cpu",
        # )
        if hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_jack_1L", "archrouting_jack_2L"]:
            for layer_id in range(config.num_hidden_layers):
                for exp_id in range(len(model.bert.encoder.layer[layer_id].other_output_experts)):
                    model.bert.encoder.layer[layer_id].other_output_experts[exp_id].LayerNorm.weight.requires_grad = False
                    model.bert.encoder.layer[layer_id].other_output_experts[exp_id].LayerNorm.bias.requires_grad = False
        if hasattr(subnet_config, "sample_expert_ids"):
            for layer_id, routes in enumerate(subnet_config.sample_expert_ids):
                if routes != 0:
                    model.bert.encoder.layer[layer_id].output.LayerNorm.weight.requires_grad = False
                    model.bert.encoder.layer[layer_id].output.LayerNorm.bias.requires_grad = False
                    model.bert.encoder.layer[layer_id].output.dense.bias.requires_grad = False
                    model.bert.encoder.layer[layer_id].output.dense.weight.requires_grad = False
                    model.bert.encoder.layer[layer_id].intermediate.dense.bias.requires_grad = False
                    model.bert.encoder.layer[layer_id].intermediate.dense.weight.requires_grad = False
                else:
                    model.bert.encoder.layer[layer_id].other_output_experts[0].LayerNorm.weight.requires_grad = False
                    model.bert.encoder.layer[layer_id].other_output_experts[0].LayerNorm.bias.requires_grad = False
                    model.bert.encoder.layer[layer_id].other_output_experts[0].dense.bias.requires_grad = False
                    model.bert.encoder.layer[layer_id].other_output_experts[0].dense.weight.requires_grad = False
                    model.bert.encoder.layer[layer_id].other_intermediate_experts[0].dense.bias.requires_grad = False
                    model.bert.encoder.layer[layer_id].other_intermediate_experts[0].dense.weight.requires_grad = False
        
        if hasattr(config, "max_experts"):
            if args.all_experts_average == 1:
                print("perform simple parameter average of all experts in a given layer...")
                subnet_copy_config = deepcopy(subnet_config)
                for key in ["max_experts", "expert_routing_type", "sample_expert_ids"]:
                    if hasattr(config, key):
                        if key == "max_experts" and args.last_expert_averaging_expert == "yes":
                            setattr(subnet_copy_config, key, getattr(config, key)-1)
                        else:
                            setattr(subnet_copy_config, key, getattr(config, key))
                print(subnet_copy_config)
                actual_model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=subnet_copy_config, ignore_mismatched_sizes=args.is_mnli_checkpoint)
                for layer_id in range(subnet_copy_config.sample_num_hidden_layers):
                    for main_expert, other_experts in [("bert.encoder.layer.<layer_id>.intermediate.dense.weight", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.intermediate.dense.bias", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.dense.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.output.dense.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.weight"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.bias")]:
                        first_expert = actual_model.state_dict()[main_expert.replace("<layer_id>", str(layer_id))].data.clone()
                        for expert_id in range(subnet_copy_config.max_experts-1):
                            first_expert = first_expert + actual_model.state_dict()[other_experts.replace("<layer_id>", str(layer_id)).replace("<expert_id>", str(expert_id))].data
                        first_expert = first_expert / float(subnet_copy_config.max_experts)
                        model.state_dict()[main_expert.replace("<layer_id>", str(layer_id))].data.copy_(first_expert)
                del first_expert
                del actual_model
            elif args.use_expert_id != 0:
                print("using expert %d instead of 0"%(args.use_expert_id))
                subnet_copy_config = deepcopy(subnet_config)
                for key in ["max_experts", "expert_routing_type", "sample_expert_ids"]:
                    if hasattr(config, key):
                        if key == "max_experts" and args.last_expert_averaging_expert == "yes":
                            setattr(subnet_copy_config, key, getattr(config, key)-1)
                        else:
                            setattr(subnet_copy_config, key, getattr(config, key))
                print(subnet_copy_config)
                actual_model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=subnet_copy_config, ignore_mismatched_sizes=args.is_mnli_checkpoint)
                for layer_id in range(subnet_copy_config.sample_num_hidden_layers):
                    for main_expert, other_experts in [("bert.encoder.layer.<layer_id>.intermediate.dense.weight", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.intermediate.dense.bias", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.dense.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.output.dense.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.weight"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.bias")]:
                        interested_expert = actual_model.state_dict()[other_experts.replace("<layer_id>", str(layer_id)).replace("<expert_id>", str(args.use_expert_id-1))].data
                        model.state_dict()[main_expert.replace("<layer_id>", str(layer_id))].data.copy_(interested_expert)
                del interested_expert
                del actual_model
            elif hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_1L", "archrouting_2L"]:
                # select the right expert for each expert layer
                actual_model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=args.is_mnli_checkpoint)
                subnet_moeconfig = deepcopy(subnet_config)
                for attr in ["max_experts", "expert_routing_type", "last_expert_averaging_expert", "sample_expert_ids"]:
                    setattr(subnet_moeconfig, attr, getattr(config, attr))
                actual_model.set_sample_config(subnet_moeconfig)
                active_arch_embed = actual_model.bert.encoder.layer[0].active_arch_embed
                selected_experts = []
                for layer_id in range(len(actual_model.bert.encoder.layer)):
                    route_prob = actual_model.bert.encoder.layer[layer_id].arch_expert(active_arch_embed)
                    route_prob_max, routes = torch.max(route_prob, dim=-1)
                    selected_experts.append(routes.item())
                    # for main_expert, other_experts in [("bert.encoder.layer.<layer_id>.intermediate.dense.weight", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.intermediate.dense.bias", "bert.encoder.layer.<layer_id>.other_intermediate_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.dense.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.weight"), ("bert.encoder.layer.<layer_id>.output.dense.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.dense.bias"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.weight", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.weight"), ("bert.encoder.layer.<layer_id>.output.LayerNorm.bias", "bert.encoder.layer.<layer_id>.other_output_experts.<expert_id>.LayerNorm.bias")]:
                    if routes != 0:
                        # interested_expert = actual_model.state_dict()[other_experts.replace("<layer_id>", str(layer_id)).replace("<expert_id>", str(routes.item()-1))].data
                        # model.state_dict()[main_expert.replace("<layer_id>", str(layer_id))].data.copy_(interested_expert)
                        model.bert.encoder.layer[layer_id].output.LayerNorm.weight.requires_grad = False
                        model.bert.encoder.layer[layer_id].output.LayerNorm.bias.requires_grad = False
                        model.bert.encoder.layer[layer_id].output.dense.bias.requires_grad = False
                        model.bert.encoder.layer[layer_id].output.dense.weight.requires_grad = False
                        model.bert.encoder.layer[layer_id].intermediate.dense.bias.requires_grad = False
                        model.bert.encoder.layer[layer_id].intermediate.dense.weight.requires_grad = False
                    else:
                        model.bert.encoder.layer[layer_id].other_output_experts[0].LayerNorm.weight.requires_grad = False
                        model.bert.encoder.layer[layer_id].other_output_experts[0].LayerNorm.bias.requires_grad = False
                        model.bert.encoder.layer[layer_id].other_output_experts[0].dense.bias.requires_grad = False
                        model.bert.encoder.layer[layer_id].other_output_experts[0].dense.weight.requires_grad = False
                        model.bert.encoder.layer[layer_id].other_intermediate_experts[0].dense.bias.requires_grad = False
                        model.bert.encoder.layer[layer_id].other_intermediate_experts[0].dense.weight.requires_grad = False

                print("selected experts per layer", selected_experts)
                # del interested_expert
                del actual_model
            #elif hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_jack_1L", "archrouting_jack_2L"]:

        print(f"Number of parameters in custom config is {millify(calculate_params_from_config(subnet_config, scaling_laws=False, add_output_emb_layer=False))}")
            
        
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

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

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    if args.subtransformer_config_path is not None:
        print('setting the subnet...')
        print(subnet_config)
        model.module.set_sample_config(subnet_config)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio*args.max_train_steps),  # args.num_warmup_steps, convert as ratio
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    '''
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )
    eval_metric = metric.compute()
    logger.info(f"before training : {eval_metric}")
    '''

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric} lr={args.learning_rate}, bs={args.per_device_train_batch_size}, ep={args.num_train_epochs}")
        print(f"epoch {epoch}: {eval_metric} lr={args.learning_rate}, bs={args.per_device_train_batch_size}, ep={args.num_train_epochs}")

    if args.skip_saving_checkpoints != "yes" and args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set # changed to matched
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")
        print(f"mnli-mm: {eval_metric}", args.learning_rate, args.per_device_train_batch_size, args.num_train_epochs)

if __name__ == "__main__":
    main()