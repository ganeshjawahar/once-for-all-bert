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
import loss
from loss import *
import wandb
from copy import deepcopy
import shutil

from xlrd import open_workbook
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from utils.module_proxy_wrapper import ModuleProxyWrapper
from tqdm.auto import tqdm
from utils import (calculate_params_from_config, millify)
from sampling import (Sampler, get_supertransformer_config)

import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
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
from torchinfo import summary

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

    # supernet configs
    parser.add_argument(
        "--mixing",
        type=str,
        required=True,
        help=f"specifies how to mix the tokens in bertlayers",
        choices=["attention", "gmlp", "fnet", "mobilebert", "bert-bottleneck"],
    )
    parser.add_argument(
        "--sampling_type",
        type=str,
        default="random",
        help=f"The sampling type for super-transformer",
        choices=["none", "naive_params", "biased_params", "random"],
    )
    parser.add_argument(
        "--sampling_rule",
        type=str,
        default="none",
        help=f"The sampling rule for sampling super-transformers",
        choices=["none", "sandwich"],
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=1,
        help=f"Number of random subtransformers to sample at each step",
    )
    parser.add_argument(
        "--inplace_distillation",
        type=int,
        default=0,
        help=f"Whether to use inplace distillation",
    )
    parser.add_argument(
        "--distillation_type",
        type=str,
        default=None,
        help=f"only used when inplace_distillation=0. logits+hiddenlastlayer, logits+attentionlastlayer, logits+hiddenlastlayer+attentionlastlayer, logits, logits+hard, logits+hard+hiddenlastlayer",
    )
    parser.add_argument(
        "--search_space_id",
        type=str,
        default=None,
        help=f"change default search space: attn_elastic, ffn_intermediate_elastic",
    )
    parser.add_argument(
        "--inplace_kd_distill_loss_weights",
        type=float,
        default=1.0,
        help=f"only useful if inplace_distillation is set",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        required=True,
        help=f"wandb entity",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="super-pretraining",
        help=f"wandb project",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="trial",
        help=f"experiment name",
    )
    parser.add_argument(
        "--wandb_suffix",
        type=str,
        default=None,
        help=f"suffix for wandb",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )

    parser.add_argument(
        "--inplace_kd_distill_loss_contrib",
        type=float,
        default=0.5,
        help=f"only useful if inplace_distillation is set",
    )

    parser.add_argument(
        "--inplace_kd_hard_loss_contrib",
        type=float,
        default=0.5,
        help=f"only useful if inplace_distillation is set",
    )

    parser.add_argument(
        "--inplace_kd_layers",
        type=str,
        default=None,
        help=f"only useful if inplace_distillation is set and distillation_type has hidden or attention",
    )

    parser.add_argument(
        "--freeze_largest_model",
        type=str,
        default="no",
        help=f"set yes to do turn off largest transformer (teacher) updates",
    )

    parser.add_argument(
        "--freeze_smallest_model",
        type=str,
        default="no",
        help=f"set yes to do turn off smallest transformer updates",
    )

    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default=None,
        help=f"path to the fixed teacher model",
    )

    parser.add_argument(
        "--validate_teacher_before_training",
        type=str,
        default="no",
        help=f"should we validate teacher before training",
    )
    
    parser.add_argument(
        "--is_mnli_checkpoint",
        type=int,
        default=0,
        help=f"if model path is a pretrained mnli checkpoint",
    )
    
    parser.add_argument(
        "--collapse_experts_before_ft",
        type=int,
        default=0,
        help=f"collapse experts before finetuning",
    )

    args = parser.parse_args()

    if args.distillation_type:
        args.distillation_type = args.distillation_type.split("+")
    if args.inplace_kd_layers:
        args.inplace_kd_layers = [int(layer) for layer in args.inplace_kd_layers.split("+")]
    else:
        args.inplace_kd_layers = [11] # defaultd last layer: todo: set it dynamically

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

    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name
        )

    if args.subtransformer_config_path is None:
        global_config = get_supertransformer_config(
            args.model_name_or_path,
            mixing=args.mixing,
            additional_random_softmaxing=False,
            random_layer_selection_probability=0.1, 
            search_space_id=args.search_space_id
        )
    else:
        global_config = get_supertransformer_config(
            args.model_name_or_path,
            mixing=args.mixing,
            additional_random_softmaxing=False,
            random_layer_selection_probability=0.1,
        )
    
    if args.distillation_type:
        # logits is always included
        if "hiddenlastlayer" in args.distillation_type:
            global_config.output_hidden_states = True # need to correct name to last_hidden_states
        if "attentionlastlayer" in args.distillation_type:
            global_config.output_attentions = True # need to correct name to attentionlastlayer
        if "tinybert" in args.distillation_type:
            global_config.output_hidden_states = True
            global_config.output_attentions = True
            global_config.add_distill_linear_layer = True # adds 
        else:
            global_config.add_distill_linear_layer = False

    # todo: remove these unused globals
    global_config.layer_drop_prob = 0.0

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    if not args.subtransformer_config_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        # model = AutoModelForSequenceClassification.from_pretrained( args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)
        global_config.num_labels = num_labels
        model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=global_config, ignore_mismatched_sizes="mnli" in args.model_name_or_path) # todo: mnli dynamic
    else:
        rb = open_workbook(args.subtransformer_config_path, formatting_info=True)
        best_config_sheet = rb.sheet_by_name("best_config")
        print("Subnet info: Model-Size=%s, Val-PPL=%s"%(best_config_sheet.cell(2, 1).value, best_config_sheet.cell(3, 1).value))
        print("Subnet info: Gene=%s"%(best_config_sheet.cell(1, 1).value))
        subnet_config = convert_to_dict(best_config_sheet.cell(4, 1).value)
        elastic_keys = eval(best_config_sheet.cell(7, 1).value)
        gene_choices = eval(best_config_sheet.cell(8, 1).value)
        gene_names = eval(best_config_sheet.cell(9, 1).value)
        elastickey2ranges = eval(best_config_sheet.cell(10, 1).value)
        print("Subnet info: Search_space_id=%s"%(best_config_sheet.cell(6, 1).value))
        print("Subnet info: elastic_keys=", elastic_keys)
        print("Subnet info: gene_choices=", gene_choices)
        print("Subnet info: gene_names=", gene_names)
        print("Subnet info: elastickey2ranges=", elastickey2ranges)
        subnet_config.num_labels = num_labels
        if args.distillation_type:
            # logits is always included
            if "hiddenlastlayer" in args.distillation_type:
                subnet_config.output_hidden_states = True # need to correct name to last_hidden_states
            if "attentionlastlayer" in args.distillation_type:
                subnet_config.output_attentions = True # need to correct name to attentionlastlayer
            if "tinybert" in args.distillation_type:
                subnet_config.output_hidden_states = True
                subnet_config.output_attentions = True
                subnet_config.add_distill_linear_layer = True # adds 
            else:
                subnet_config.add_distill_linear_layer = False
        
        if args.collapse_experts_before_ft == 0:
            config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
            if hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_jack_1L", "archrouting_jack_2L", "archrouting_1L", "archrouting_2L", "neuronrouting_jack_2L"]:
                for attr in ["max_experts", "expert_routing_type", "last_expert_averaging_expert", "sample_expert_ids", "fixed_hypernet_input", "hypernet_hidden_size", "hypernet_input_format"]:
                    if hasattr(config, attr):
                        setattr(subnet_config, attr, getattr(config, attr))

            model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=subnet_config, ignore_mismatched_sizes="mnli" in args.model_name_or_path) # todo: mnli dynamic
            print(f"Number of parameters in custom config is {millify(calculate_params_from_config(subnet_config, scaling_laws=False, add_output_emb_layer=False))}")
            # config.hidden_dropout_prob = 0.1

            if hasattr(config, "expert_routing_type") and config.expert_routing_type in ["archrouting_jack_1L", "archrouting_jack_2L", "neuronrouting_jack_2L"]:
                for layer_id in range(subnet_config.sample_num_hidden_layers):
                    for exp_id in range(len(model.bert.encoder.layer[layer_id].other_output_experts)):
                        # set layernorm weight to False
                        model.bert.encoder.layer[layer_id].other_output_experts[exp_id].LayerNorm.weight.requires_grad = False
                        model.bert.encoder.layer[layer_id].other_output_experts[exp_id].LayerNorm.bias.requires_grad = False
        else:
            if hasattr(subnet_config, "max_experts"):
                delattr(subnet_config, "max_experts")
                delattr(subnet_config, "expert_routing_type")
            model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=subnet_config, ignore_mismatched_sizes="mnli" in args.model_name_or_path) # todo: mnli dynamic
            model.set_sample_config(subnet_config)
            print(f"Number of parameters in custom config is {millify(calculate_params_from_config(subnet_config, scaling_laws=False, add_output_emb_layer=False))}")

            if args.is_mnli_checkpoint == 0:
                print("config for the actual model...")
                actual_config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
                print(actual_config)

                print("generate arch. expert parameters in a given layer...")
                actual_model = custom_bert.BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=actual_config, ignore_mismatched_sizes=args.is_mnli_checkpoint)
                subnet_moeconfig = deepcopy(subnet_config)
                for attr in ["max_experts", "expert_routing_type", "last_expert_averaging_expert", "sample_expert_ids", "fixed_hypernet_input", "hypernet_hidden_size", "hypernet_input_format"]:
                    if hasattr(actual_config, attr):
                        setattr(subnet_moeconfig, attr, getattr(actual_config, attr))
                actual_model.set_sample_config(subnet_moeconfig)
                print("config for the subnet model...")
                print(subnet_moeconfig)

                active_arch_embed = actual_model.bert.encoder.layer[0].active_arch_embed
                print("active_arch_embed", active_arch_embed)
                for layer_id in range(actual_config.sample_num_hidden_layers):
                    if actual_config.expert_routing_type == "archrouting_jack_1L" or actual_config.expert_routing_type == "archrouting_jack_2L":
                        route_prob = actual_model.bert.encoder.layer[layer_id].arch_expert(active_arch_embed)
                        route_prob_max, routes = torch.max(route_prob, dim=-1)
                        intermediate_weights = route_prob[0] * actual_model.bert.encoder.layer[layer_id].intermediate.dense.samples["weight"]
                        intermediate_bias = route_prob[0] * actual_model.bert.encoder.layer[layer_id].intermediate.dense.samples["bias"]
                        layer_output_weights = route_prob[0] * actual_model.bert.encoder.layer[layer_id].output.dense.samples["weight"]
                        layer_output_bias = route_prob[0] * actual_model.bert.encoder.layer[layer_id].output.dense.samples["bias"]
                        for expert_id in range(actual_config.max_experts-1):
                            intermediate_weights = intermediate_weights + (route_prob[expert_id+1] * actual_model.bert.encoder.layer[layer_id].other_intermediate_experts[expert_id].dense.samples["weight"])
                            intermediate_bias = intermediate_bias + (route_prob[expert_id+1] * actual_model.bert.encoder.layer[layer_id].other_intermediate_experts[expert_id].dense.samples["bias"])
                            layer_output_weights = layer_output_weights + (route_prob[expert_id+1] * actual_model.bert.encoder.layer[layer_id].other_output_experts[expert_id].dense.samples["weight"])
                            layer_output_bias = layer_output_bias + (route_prob[expert_id+1] * actual_model.bert.encoder.layer[layer_id].other_output_experts[expert_id].dense.samples["bias"])
                        model.state_dict()["bert.encoder.layer.<layer_id>.intermediate.dense.weight".replace("<layer_id>", str(layer_id))].data[0:intermediate_weights.size(0), 0:intermediate_weights.size(1)] = intermediate_weights
                        model.state_dict()["bert.encoder.layer.<layer_id>.intermediate.dense.bias".replace("<layer_id>", str(layer_id))].data[0:intermediate_bias.size(0)] = intermediate_bias
                        model.state_dict()["bert.encoder.layer.<layer_id>.output.dense.weight".replace("<layer_id>", str(layer_id))].data.data[0:layer_output_weights.size(0), 0:layer_output_weights.size(1)] = layer_output_weights
                        model.state_dict()["bert.encoder.layer.<layer_id>.output.dense.bias".replace("<layer_id>", str(layer_id))].data[0:layer_output_bias.size(0)] = layer_output_bias
                    elif actual_config.expert_routing_type in ["neuronrouting_jack_2L"]:
                        fc1_expert_out = actual_model.bert.encoder.layer[layer_id].arch_expert_fc1(active_arch_embed)
                        fc1_expert_out = fc1_expert_out.view(-1, actual_config.max_experts)
                        fc1_expert_out = torch.nn.Softmax(dim=-1)(fc1_expert_out)
                        fc2_expert_out = actual_model.bert.encoder.layer[layer_id].arch_expert_fc2(active_arch_embed)
                        fc2_expert_out = fc2_expert_out.view(-1, actual_config.max_experts)
                        fc2_expert_out = torch.nn.Softmax(dim=-1)(fc2_expert_out)
                        intermediate_weights = actual_model.bert.encoder.layer[layer_id].intermediate.dense.samples["weight"] * fc1_expert_out[:, 0].view(-1,1)
                        intermediate_bias = actual_model.bert.encoder.layer[layer_id].intermediate.dense.samples["bias"] * fc1_expert_out[:, 0]
                        layer_output_weights = actual_model.bert.encoder.layer[layer_id].output.dense.samples["weight"] * fc2_expert_out[:, 0].view(-1,1)
                        layer_output_bias = actual_model.bert.encoder.layer[layer_id].output.dense.samples["bias"] * fc2_expert_out[:, 0]
                        for expert_id in range(actual_config.max_experts-1):
                            intermediate_weights = intermediate_weights + (actual_model.bert.encoder.layer[layer_id].other_intermediate_experts[expert_id].dense.samples["weight"] * fc1_expert_out[:, expert_id+1].view(-1,1))
                            intermediate_bias = intermediate_bias + (actual_model.bert.encoder.layer[layer_id].other_intermediate_experts[expert_id].dense.samples["bias"] * fc1_expert_out[:, expert_id+1] )
                            layer_output_weights = layer_output_weights + (actual_model.bert.encoder.layer[layer_id].other_output_experts[expert_id].dense.samples["weight"] * fc2_expert_out[:, expert_id+1].view(-1,1))
                            layer_output_bias = layer_output_bias + (actual_model.bert.encoder.layer[layer_id].other_output_experts[expert_id].dense.samples["bias"] * fc2_expert_out[:, expert_id+1] )
                        model.state_dict()["bert.encoder.layer.<layer_id>.intermediate.dense.weight".replace("<layer_id>", str(layer_id))].data[0:intermediate_weights.size(0), 0:intermediate_weights.size(1)] = intermediate_weights
                        model.state_dict()["bert.encoder.layer.<layer_id>.intermediate.dense.bias".replace("<layer_id>", str(layer_id))].data[0:intermediate_bias.size(0)] = intermediate_bias
                        model.state_dict()["bert.encoder.layer.<layer_id>.output.dense.weight".replace("<layer_id>", str(layer_id))].data.data[0:layer_output_weights.size(0), 0:layer_output_weights.size(1)] = layer_output_weights
                        model.state_dict()["bert.encoder.layer.<layer_id>.output.dense.bias".replace("<layer_id>", str(layer_id))].data[0:layer_output_bias.size(0)] = layer_output_bias
                del actual_model

    # print(subnet_config)
    print(summary(model, depth=4, verbose=0))

    if args.teacher_model_path is not None:
        #teacher_config = deepcopy(global_config)
        #if args.distillation_type:
        #    teacher_config.add_distill_linear_layer = False
        teacher_config = AutoConfig.from_pretrained(args.teacher_model_path, num_labels=num_labels, finetuning_task=args.task_name,  ignore_mismatched_sizes="mnli" in args.teacher_model_path)
        # logits is always included
        if "hiddenlastlayer" in args.distillation_type:
            teacher_config.output_hidden_states = True # need to correct name to last_hidden_states
        if "attentionlastlayer" in args.distillation_type:
            teacher_config.output_attentions = True # need to correct name to attentionlastlayer
        if "tinybert" in args.distillation_type:
            teacher_config.output_hidden_states = True
            teacher_config.output_attentions = True
        # todo: make the following dynamic
        '''
        for param, def_value in [("mixing", "attention"), ("normalization_type", "layer_norm"), ("sample_hidden_size", [768 for i in range(12)]), ("sample_intermediate_size", [3072 for i in range(12)]), ("sample_num_attention_heads", [12 for i in range(12)]), ("sample_num_hidden_layers", 12)]:
            if not hasattr(teacher_config, param):
                setattr(teacher_config, param, def_value)
        print(teacher_config)
        '''

        if os.path.exists(args.teacher_model_path) and hasattr(teacher_config, "mixing"):
            teacher_model = custom_bert.BertForSequenceClassification.from_pretrained(args.teacher_model_path, config=teacher_config)
        else:
            teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_path, config=teacher_config)
        
        teacher_model.eval()
        print('setting teachers requires_grad to False')
        # hack to use accelerator to prepare teacher model
        i = 0
        for p in teacher_model.parameters(): 
            if i != 0:
                p.requires_grad = False
            i += 1

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
    if args.teacher_model_path is not None:
        teacher_model = accelerator.prepare(teacher_model) # not needed when teacher doesn't have any parameter that requires a gradient

    '''
    if (
        accelerator.distributed_type == DistributedType.MULTI_GPU
        or accelerator.distributed_type == DistributedType.TPU
    ):
        # forward missing getattr and state_dict/load_state_dict to orig model
        model = ModuleProxyWrapper(model)
        if args.teacher_model_path is not None:
            teacher_model = ModuleProxyWrapper(teacher_model)
    '''

    if args.subtransformer_config_path is not None:
        print('setting the subnet...')
        print(subnet_config)
        model.module.set_sample_config(subnet_config)
    else:
        model.module.set_sample_config(global_config, drop_layers=False)
    if args.teacher_model_path is not None and os.path.exists(args.teacher_model_path) and hasattr(teacher_config, "mixing"):
        teacher_model.module.set_sample_config(teacher_config)

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
    if accelerator.is_main_process:
        wandb.watch(model, log=None, log_freq=100)

    sampler = Sampler(
        args.sampling_type,
        args.sampling_rule,
        args.mixing,
        global_config,
        search_space_id=args.search_space_id
    )

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
    '''

    if args.validate_teacher_before_training == "yes":
        for step, batch in enumerate(eval_dataloader):
            outputs = teacher_model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
        supernet_eval_metric = metric.compute()
        print("teacher model", supernet_eval_metric)
        sys.exit(0)

    seed = -1
    completed_epochs = 0
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            seed += 1
            if (
                (args.sampling_type != "none")
                and step % args.gradient_accumulation_steps == 0
            ):
                config_dict = sampler.sample_subtransformer(
                    randomize=True,
                    rand_seed=seed,
                    pop_size=args.pop_size
                )
                super_config_small = config_dict["smallest_subtransformer"]
                # list of random subtransformers with len pop_size
                super_configs = config_dict["random_subtransformers"]
            track_loss = step % args.logging_steps == 0 and step > 0

            ### Applying Sandwich Rule ###
            if args.sampling_rule == "sandwich" or args.inplace_distillation:
                if args.sampling_rule == "sandwich":
                    ## Sample Supertransformer
                    if args.teacher_model_path is not None:
                        outputs = teacher_model(**batch)
                    else:
                        model.module.set_sample_config(global_config, drop_layers=True)
                        outputs = model(**batch)
                        loss = outputs.loss
                        teacher_mlm_loss = loss
                        teacher_mlm_loss = teacher_mlm_loss / args.gradient_accumulation_steps
                        accelerator.backward(teacher_mlm_loss)

                    if args.inplace_distillation:

                        teacher_info = {}
                        if "hiddenlastlayer" in args.distillation_type or "tinybert" in args.distillation_type:
                            teacher_info["teacher_hidden_states"] = outputs.hidden_states

                        if "attentionlastlayer" in args.distillation_type or "tinybert" in args.distillation_type:
                            teacher_info["teacher_attention_maps"] = outputs.attentions
                        
                        if "logits" in args.distillation_type:
                            teacher_info["teacher_logits"] = outputs.logits.detach()
                elif args.inplace_distillation and args.freeze_largest_model == "yes" and args.freeze_smallest_model == "yes":
                    # need teacher logits
                    if args.teacher_model_path is None:
                        model.eval()
                        model.module.set_sample_config(global_config, drop_layers=True)
                        outputs = model(**batch)
                        teacher_info = {}
                        if "hiddenlastlayer" in args.distillation_type:
                            teacher_info["teacher_hidden_states"] = outputs.hidden_states
                        if "attentionlastlayer" in args.distillation_type:
                            teacher_info["teacher_attention_maps"] = outputs.attentions
                        if "logits" in args.distillation_type:
                            teacher_info["teacher_logits"] = outputs.logits.detach()
                        model.train()
                    else:
                        with torch.no_grad():
                            outputs = teacher_model(**batch)
                        teacher_info = {}
                        if "hiddenlastlayer" in args.distillation_type or "tinybert" in args.distillation_type:
                            teacher_info["teacher_hidden_states"] = outputs.hidden_states
                        if "attentionlastlayer" in args.distillation_type or "tinybert" in args.distillation_type:
                            teacher_info["teacher_attention_maps"] = outputs.attentions
                        if "logits" in args.distillation_type:
                            teacher_info["teacher_logits"] = outputs.logits.detach()

                if args.sampling_rule == "sandwich":
                    ## Sample Smallest Subtransformer
                    
                    model.module.set_sample_config(super_config_small, drop_layers=True)
                    outputs = model(**batch) #, use_soft_loss=args.inplace_distillation)
                    loss = outputs.loss

                    if args.inplace_distillation:

                        (
                            smallest_student_loss,
                            smallest_student_losses_dict,
                        ) = compute_student_loss(
                            outputs,
                            teacher_info,
                            args,
                            track_layerwise_loss=track_loss,
                        )
                    else:
                        smallest_student_loss = loss
                        smallest_student_loss = (
                            smallest_student_loss / args.gradient_accumulation_steps
                        )
                    accelerator.backward(smallest_student_loss)
                    
            for idx in range(args.pop_size):

                if args.sampling_type != "none":
                    super_config = super_configs[idx]
                    
                    model.module.set_sample_config(super_config, drop_layers=True)

                outputs = model(**batch) #, use_soft_loss=args.inplace_distillation)
                loss = outputs.loss

                if args.inplace_distillation:

                    (
                        sampled_student_loss,
                        sampled_student_losses_dict,
                    ) = compute_student_loss(
                        outputs,
                        teacher_info,
                        args,
                        track_layerwise_loss=track_loss,
                    )
                else:
                    sampled_student_loss = loss / args.gradient_accumulation_steps

                accelerator.backward(sampled_student_loss)
            
            # cleanup
            if args.inplace_distillation or args.distillation_type:
                del teacher_info
                del outputs

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if accelerator.is_main_process and track_loss:
                if not args.inplace_distillation:
                    if args.sampling_rule == "sandwich":
                        wandb.log(
                            {
                                "Supertransformer mlm loss": teacher_mlm_loss.item(),
                                "Smallest mlm loss": smallest_student_loss.item(),
                                "Subtransformer mlm loss": sampled_student_loss.item(),
                            }
                        )
                    else:
                        wandb.log(
                            {
                                "Subtransformer mlm loss": sampled_student_loss.item(),
                            }
                        )
                else:
                    log = {}
                    if args.freeze_largest_model == "no":
                        log["Supertransformer Teacher mlm loss"] = teacher_mlm_loss.item()
                    if args.freeze_smallest_model == "no":
                        log["Smallest Student mlm loss"] =  smallest_student_losses_dict["student_mlm_loss"]
                    if args.freeze_largest_model == "yes" and args.freeze_smallest_model == "yes":
                        log["Subtransformer mlm loss"] = sampled_student_losses_dict["student_mlm_loss"]
                    if "logits" in args.distillation_type:
                        if args.freeze_smallest_model == "no":
                            log["Smallest Student distill loss"] = smallest_student_losses_dict["student_distill_loss"]
                        log["Subtransformer Student distill loss"] = sampled_student_losses_dict["student_distill_loss"]
                    if "hiddenlastlayer" in args.distillation_type:
                        if args.freeze_smallest_model == "no":
                            log["Smallest Student hidden loss"] = smallest_student_losses_dict["student_hidden_loss"]
                        log["Subtransformer Student hidden loss"] = sampled_student_losses_dict["student_hidden_loss"]
                    if "attentionlastlayer" in args.distillation_type:
                        if args.freeze_smallest_model == "no":
                            log["Smallest Student attention loss"] = smallest_student_losses_dict["student_attention_loss"]
                        log["Subtransformer Student attention loss"] = sampled_student_losses_dict["student_attention_loss"]
                    wandb.log(log)
            
            if accelerator.is_main_process:
                wandb.log({"epochs": epoch})

            if completed_steps >= args.max_train_steps:
                break

        # change to supertransformer config
        if args.subtransformer_config_path is not None and args.sampling_type == "none":
            model.module.set_sample_config(subnet_config, drop_layers=False)
        else:
            model.module.set_sample_config(global_config, drop_layers=False)
        model.eval()

        # valid acc for supernet
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
        supernet_eval_metric = metric.compute()
        print(supernet_eval_metric)
        wandb_log = {"SuperTransformer Val Accuracy": supernet_eval_metric['accuracy'] if "accuracy" in supernet_eval_metric else supernet_eval_metric['matthews_correlation'], "task": args.task_name, "epoch": epoch, "num_train_epochs": args.num_train_epochs}

        if args.sampling_type != "none":
            # valid acc for subnet
            config_dict = sampler.sample_subtransformer(
                randomize=True,
                rand_seed=seed,
                pop_size=args.pop_size,
                v1_small=True # todo: make this dynamic based on search space
            )
            super_config_small = config_dict["smallest_subtransformer"]
            model.module.set_sample_config(super_config_small, drop_layers=False)
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )
            smallestnet_eval_metric = metric.compute()
            wandb_log["SmallestTransformer Val Accuracy"] = smallestnet_eval_metric['accuracy']  if "accuracy" in smallestnet_eval_metric else smallestnet_eval_metric['matthews_correlation']
        
        if accelerator.is_main_process:
            print(wandb_log)
            wandb.log(wandb_log)
        #logger.info(
        #    f"epoch {epoch}: SuperTransformer Val Accuracy:  {supernet_eval_metric['accuracy']:.2f}"
        #)
        completed_epochs += 1

    if args.skip_saving_checkpoints != "yes" and args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    if args.task_name == "mnli":
        accelerator.wait_for_everyone()
        best_ckpt_path = os.path.join(args.output_dir, "mnli_best")
        best_acc = None
        if os.path.exists(best_ckpt_path) and os.path.exists(best_ckpt_path + "/score"):
            for line in open(best_ckpt_path + "/score"):
                best_acc = float(line.strip())
                break
        if best_acc is None or best_acc < supernet_eval_metric['accuracy']:
            # if os.path.exists(best_ckpt_path):
            #    shutil.rmtree(best_ckpt_path)
            os.makedirs(best_ckpt_path, exist_ok=True)
            w = open(best_ckpt_path + "/score", 'w')
            w.write("%s\n"%(str(supernet_eval_metric['accuracy'])))
            w.close()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_ckpt_path, save_function=accelerator.save)

    logger.info(f"Training completed. Find your checkpoints at {args.output_dir}")


if __name__ == "__main__":
    main()

