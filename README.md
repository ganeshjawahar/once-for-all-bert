# Mixture-of-Supernets: Improving Weight-Sharing Supernet Training with Architecture-Routed Mixture-of-Experts

This repository contains code used in Mixture-of-Supernet (MoS) work. This repository builds on [SuperShaper](https://github.com/iitm-sysdl/SuperShaper).

## Quick Setup

### (1) Install
Run the following commands to install MoS:
```
pip install -r requirements.txt
```

### (2) Set the paths to code, data and output
Set the paths in the `aws/start_jobs.py` file

#### (2a) Path to model files and other outputs
Set the path to where the model files and other outputs need to be stored:

    def get_experiments_dir():
        return "/fsx/ganayu/experiments/supershaper"
        

#### (2b) Path to code
Set the path to parent directory of the codebase:

    def get_code_dir():
        return "/fsx/ganayu/code/SuperShaper"
    

#### (2c) Path to preprocessed data
Download preprocessed data from [here](https://1drv.ms/u/s!AlflMXNPVy-wgpEBtZFHfRkp9IMm3w?e=ujMhfG) and extract. Set the path to extracted preprocessed data:

    def dataset_factory(name):
        datasets = {}
        datasets["wikibooks_acabert_bertbaseuncased_128len"] = "/fsx/ganayu/data/academic_bert_dataset/final_bert_preproc_128" 
        return datasets[name]

#### (2d) Path to finetuned BERT-Base GLUE models 
Download finetuned models from [here](https://1drv.ms/u/s!AlflMXNPVy-wgpEE2i9FvCUTQkpN6Q?e=IILAmX) and extract. Set the path to extracted finetuned models:

    def task_specific_trained_teacher(name):
        teacher_models = {}

        teacher_models["bertbase"] = {"cola": "/fsx/ganayu/experiments/supershaper/aug25_finetune_googlebert_mnli_cola_ckptneeded/32_bertbase_cola_2e-5_16_4", "mnli": "/fsx/ganayu/experiments/supershaper/aug25_finetune_googlebert_mnli_cola_ckptneeded/6_bertbase_mnli_3e-5_16_2", "sst2": "/fsx/ganayu/experiments/supershaper/aug25_finetune_googlebert_mnli_cola_ckptneeded/43_bertbase_sst2_3e-5_16_3", "qnli": "/fsx/ganayu/experiments/supershaper/aug25_finetune_googlebert_mnli_cola_ckptneeded/60_bertbase_qnli_3e-5_16_2", "qqp": "/fsx/ganayu/experiments/supershaper/aug25_finetune_googlebert_mnli_cola_ckptneeded/77_bertbase_qqp_5e-5_32_4", "mrpc": "/fsx/ganayu/experiments/supershaper/aug25_finetune_googlebert_mrpc_rte_stsb_ckptneeded/1_bertbase_mrpc_5e-5_16_3", "rte": "/fsx/ganayu/experiments/supershaper/aug25_finetune_googlebert_mrpc_rte_stsb_ckptneeded/22_bertbase_rte_5e-5_32_3"}

        return teacher_models[name]
        

### (3) Pretrain MoS supernet
Pretrain MoS supernet by adding the following command in  `aws/start_jobs.py`:


    script_creator(get_experiments_dir() + "/nov10_neuronrouting_jack_2L", [ {"exp_name": "neuron", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "nov10_neuronrouting_jack_2L", "tokenized_c4_dir": dataset_factory("wikibooks_acabert_bertbaseuncased_128len"), "tokenizer_name": "bert-base-uncased", "max_experts": 2, "hypernet_hidden_size": 128, "pop_size": 1, "expert_routing_type": "neuronrouting_jack_2L", "max_train_steps": 125000})}]}  ], time_in_mins=10000, wandb="online")

where,
* `parent_experiment_name` - parent experiment name (e.g., `nov10_neuronrouting_jack_2L`)
* `exp_name` - child experiment name (e.g., `neuron`)
* `pyfile` - supernet pretraining script (e.g., `train_mlm.py`)
* `params` - parameters for the run
* `experiment_name` - experiment name that can be ignored
* `tokenized_c4_dir` - path to preprocessed directory (e.g., `dataset_factory("wikibooks_acabert_bertbaseuncased_128len")`)
* `tokenizer_name` - tokenizer name (e.g., `bert-base-uncased`)
* `max_experts` - number of expert weights (e.g., `2`)
* `hypernet_hidden_size` - router hidden dimension (e.g., `128`)
* `pop_size` - number of random networks to sample (e.g., `1`)
* `expert_routing_type` - MoS variant (e.g., `archrouting_jack_2L` for layer-wise MoS, `neuronrouting_jack_2L` for neuron-wise MoS)
* `max_train_steps` - maximum number of train steps (e.g., `125000`)
* `time_in_mins` - maximum number of minutes for the job to run (e.g., `10000`)

The best checkpoint is at `<parent_experiment_name>/<exp_name>/checkpoint_best.pt`. Pre-trained supernets can be download from [here](https://1drv.ms/u/s!AlflMXNPVy-wgpEDtGrTgezWmmU2lw?e=3Aa1kE).

### (3) Run evolutionary search
Run evolutionary search by adding the following command in  `aws/start_jobs.py`:

    script_creator(get_experiments_dir() + "/nov21_neuronmoe_50M", [ {"exp_name": name, "runs": [ {"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/nov10_neuronrouting_jack_2L/neuron/best_model", "mixing": "bert-bottleneck", "params_constraints": constraint, "data_dir": dataset_factory("wikibooks_acabert_bertbaseuncased_128len"), "max_experts": 2, "expert_routing_type": "neuronrouting_jack_2L", "last_expert_averaging_expert": "no", "hypernet_hidden_size": 128})}]} for name, constraint in [("50M", "45000000,50000000")] ], time_in_mins=8000, wandb="no", num_gpus=4, generate_port=True)

where,
* `nov21_neuronmoe_50M` - parent experiment name
* `exp_name` - child experiment name
* `pyfile` - evolutionary search script (e.g., `evo_ours.py`)
* `params` - parameters for the run (e.g., `modify_config_and_to_string(config_factory("supernetbasev3.evosearch")`)
* `supernet_ckpt_dir` - path to best supernet checkpoint (e.g., `get_experiments_dir() + "/nov10_neuronrouting_jack_2L/neuron/best_model"`)
* `params_constraints` - minimum and maximum parameter size in csv format (e.g., `45000000,50000000`)
* `data_dir` - path to preprocessed directory (e.g., `dataset_factory("wikibooks_acabert_bertbaseuncased_128len")`)
* `max_experts` - number of expert weights (e.g., `2`)
* `expert_routing_type` - MoS variant (e.g., `archrouting_jack_2L` for layer-wise MoS, `neuronrouting_jack_2L` for neuron-wise MoS)
* `hypernet_hidden_size` - router hidden dimension (e.g., `128`)
* `time_in_mins` - maximum number of minutes for the job to run (e.g., `10000`)
* `num_gpus` - number of GPUs (e.g., `4`)

The config for the best architecture is at `<parent_experiment_name>/<exp_name>/evo_results_29.xls`.

### (4) Finetune best subnet on downstream task
Run finetuning best subnet on all GLUE downstream tasks by adding the following command in  `aws/start_jobs.py`:

    for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
        script_creator(get_experiments_dir() + "/nov22_finetune_neuronmoe_50M_%s"%exp_name, create_finetuning_experiments_standalone_vs_supernet_v3(num_gpus=2, tasks=tasks, glue_config="supernetbasev3.standard.distill.supernet_finetune", sweep=get_finetuning_sweeps("bert"), models=[ {"exp_name": exp_name, "model_name_or_path": get_experiments_dir() + "/nov10_neuronrouting_jack_2L/neuron/best_model", "subtransformer_config_path": get_experiments_dir() + "/nov21_neuronmoe_50M/50M/evo_results_29.xls", "tokenizer_name": "bert-base-uncased", "mixing": "bert-bottleneck"} ], kd_teacher=task_specific_trained_teacher("bertbase"), kd_config=finetuning_kd_config("std_logits")), time_in_mins=15000, wandb="offline", num_gpus=2, generate_port=True)
        

where,
* `nov22_finetune_neuronmoe_50M_` - parent experiment name
* `exp_name` - child experiment name
* `num_gpus` - number of GPUs (e.g., `2`)
* `tasks` - set of GLUE tasks (e.g., `["mnli", "mrpc", "rte"]`)
* `glue_config` - finetuning settings (e.g., `"supernetbasev3.standard.distill.supernet_finetune"`)
* `sweep` - sweep for finetuning (e.g., `get_finetuning_sweeps("bert")`)
* `model_name_or_path` - path to the best supernet checkpoint (e.g., `get_experiments_dir() + "/nov10_neuronrouting_jack_2L/neuron/best_model"`)
* `subtransformer_config_path` - path to the best subnet config (e.g., `get_experiments_dir() + "/nov21_neuronmoe_50M/50M/evo_results_29.xls"`)
* `tokenizer_name` - tokenizer name (e.g., `bert-base-uncased`)
* `kd_teacher` - path to the finetuned BERT-base models on GLUE tasks (e.g., `task_specific_trained_teacher("bertbase")`)
* `kd_config` - config for distillation (e.g., `finetuning_kd_config("std_logits")`)
* `time_in_mins` - maximum number of minutes for the job to run (e.g., `10000`)
* `num_gpus` - number of GPUs (e.g., `4`)


### (5) Re-pretrain best subnet
Re-pretrain best subnet by adding the following command in `aws/start_jobs.py`:

    script_creator(get_experiments_dir() + "/nov22_standalone_neuronmoe_50M", [ {"exp_name": "stand_50M", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "nov22_standalone_neuronmoe_50M", "max_train_steps": 125000, "sampling_type": "none", "num_warmup_steps": 0, "sampling_rule": "none", "model_name_or_path": get_experiments_dir() + "/nov10_neuronrouting_jack_2L/neuron/best_model", "subtransformer_config_path": get_experiments_dir() + "/nov21_neuronmoe_50M/50M/evo_results_29.xls", "tokenized_c4_dir": dataset_factory("wikibooks_acabert_bertbaseuncased_128len"), "tokenizer_name": "bert-base-uncased", "mixing": "bert-bottleneck"})}]} ], time_in_mins=8000, wandb="online", generate_port=True, num_gpus=8)

where,
* `nov22_standalone_neuronmoe_50M` - parent experiment name
* `exp_name` - child experiment name
* `pyfile` - supernet re-pretraining script (e.g., `train_mlm.py`)
* `params` - parameters for the run
* `experiment_name` - experiment name (e.g., `nov22_standalone_neuronmoe_50M`)
* `tokenized_c4_dir` - path to preprocessed directory (e.g., `dataset_factory("wikibooks_acabert_bertbaseuncased_128len")`)
* `tokenizer_name` - tokenizer name (e.g., `bert-base-uncased`)
* `max_train_steps` - maximum number of train steps (e.g., `125000`)
* `time_in_mins` - maximum number of minutes for the job to run (e.g., `8000`)
* `num_gpus` - number of GPUs (e.g., `8`)
* `subtransformer_config_path` - path to the subnet configuration file (e.g., `<parent_experiment_name>/<exp_name>/evo_results_29.xls`)

The best checkpoint is at `<parent_experiment_name>/<exp_name>/checkpoint_best.pt`.









