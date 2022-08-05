'''
utilities to start jobs in AWS
'''

import sys, os, json, glob, shutil, random

def script_creator(output_folder, commands, time_in_mins=15000, num_gpus=8, code_dir="/fsx/ganayu/code/SuperShaper", wandb="online", wandb_entity="ganayu", wandb_project="effbert", generate_port=False, part_size=-1):
    if os.path.exists(output_folder):
        print("%s already exists"%(output_folder))
        sys.exit(0)
    print("ensure you are in login node and 'basic' conda environment")
    # create output folder
    os.makedirs(output_folder)
    # copy the current codebase
    cur_code_dir = output_folder + "/code"
    shutil.copytree(code_dir, cur_code_dir)
    # create environment script
    cur_envscript = output_folder + "/custom_config.yaml"
    w_envscript = open(cur_envscript, "w")
    for line in open(code_dir + "/default_config.yaml"):
        line = line.strip()
        if "num_processes" in line:
            line = "num_processes: %d"%num_gpus
        w_envscript.write(line.strip()+"\n")
    w_envscript.close()
    # create job scripts
    cur_bsz, cur_master_commands = 0, []
    master_shell_f = output_folder + "/master%d.sh"%len(cur_master_commands)
    cur_master_commands.append('bash %s'%(master_shell_f))
    w_master = open(master_shell_f, "w")
    for command in commands:
        cur_out_dir = output_folder + "/" + command["exp_name"]
        os.makedirs(cur_out_dir, exist_ok=True)
        cur_sbatch_f = cur_out_dir + "/sbatch.sh"
        cur_shell_f = cur_out_dir + "/shell.sh"
        w_sbatch = open(cur_sbatch_f, 'w')
        w_sbatch.write("#!/bin/bash\n")
        w_sbatch.write("#SBATCH --job-name=%s\n"%command["exp_name"])
        w_sbatch.write("#SBATCH --output=%s\n"%(cur_out_dir + "/sys.out"))
        w_sbatch.write("#SBATCH --error=%s\n"%(cur_out_dir + "/sys.err"))
        w_sbatch.write("#SBATCH --partition=a100\n")
        w_sbatch.write("#SBATCH --nodes=1\n")
        w_sbatch.write("#SBATCH --ntasks-per-node=1\n")
        w_sbatch.write("#SBATCH --cpus-per-task=10\n")
        w_sbatch.write("#SBATCH --gres=gpu:%d\n"%num_gpus)
        w_sbatch.write("#SBATCH --time %d\n"%time_in_mins)
        w_sbatch.write("module purge\n")
        w_sbatch.write("echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID\n")
        w_sbatch.write("nvidia-smi\n")
        # w_sbatch.write("wandb %s\n"%(wandb))
        w_sbatch.write("srun --label %s"%cur_shell_f)
        w_sbatch.close()
        w_shell = open(cur_shell_f, "w")
        w_shell.write("#!/bin/bash\n")
        if wandb != "no":
            w_shell.write("wandb %s\n"%(wandb))
        w_shell.write("TOKENIZERS_PARALLELISM=false\n")
        w_shell.write("cd %s\n"%cur_code_dir)
        for run in command["runs"]:
            if "<<OUTPUT_DIR>>" in run["params"]:
                run["params"] = run["params"].replace("<<OUTPUT_DIR>>", cur_out_dir)      
            if wandb != "no":
                w_shell.write("WANDB_MODE=%s "%wandb)
            w_shell.write("accelerate launch --config_file %s"%(cur_envscript))
            if generate_port:
                w_shell.write(" --main_process_port %d"%(random.randint(17000, 22000)))
            w_shell.write(" %s %s"%(run["pyfile"], run["params"]))
            if wandb != "no":
                w_shell.write(" --wandb_suffix %s --wandb_entity %s --wandb_project %s"%(command["exp_name"], wandb_entity, wandb_project))
            w_shell.write("\n")
        w_shell.close()
        w_master.write("sbatch %s\n"%(cur_sbatch_f))
        os.system("chmod 777 %s/*.sh"%(cur_out_dir))
        cur_bsz += 1
        if part_size != -1 and cur_bsz % part_size == 0:
            w_master.close()
            cur_bsz = 0
            master_shell_f = output_folder + "/master%d.sh"%len(cur_master_commands)
            cur_master_commands.append('bash %s'%(master_shell_f))
            w_master = open(master_shell_f, "w")
    w_master.close()
    os.system("chmod 777 %s/*.sh"%(output_folder))
    for master_command in cur_master_commands:
        print(master_command)

def dataset_factory(name):
    datasets = {}
    
    # bert-pretraining data
    datasets["wikibooks"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final"
    datasets["wikibooks_dummy"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy"
    datasets["wikibooks_tokenized"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final_tokenized"
    datasets["wikibooks_dummy_tokenized"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy_tokenized"
    datasets["wikibooks_graphcore_128len"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_graphcore_128"
    datasets["wikibooks_graphcore_128len_next_sentence_label_removed_w_splits"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_graphcore_128_next_sentence_label_removed_w_splits"


    # c4-pretraining data
    datasets["c4"] = "/fsx/ganayu/data/supershaper_pretraining_data/c4_datasets_cleaned"

    return datasets[name]

def config_factory(name):
    configs = {}
    # std. supershaper
    configs["supershaper.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 128, "mixing": "bert-bottleneck", "max_train_steps": 175214, "c4_dir": dataset_factory("c4"), "model_name_or_path": "bert-base-cased", "sampling_type": "random", "sampling_rule": "sandwich", "learning_rate": 2e-5, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 1, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1}
    configs["supershaper.standard.train_glue"] = {"learning_rate": 5e-05, "mixing": "bert-bottleneck", "model_name_or_path": "", "num_train_epochs": 10, "per_device_train_batch_size": 4, "sampling_type": "none", "task": "", "subtransformer_config_path": "", "output_dir": "<<OUTPUT_DIR>>"}

    # phase-2 changes (didn't work)
    # 125000*2048 == 1000000*256
    configs["bertbase.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 512, "mixing": "attention", "max_train_steps": 125000, "c4_dir": dataset_factory("wikibooks"), "model_name_or_path": "bert-base-cased", "sampling_type": "none", "sampling_rule": "none", "learning_rate": 1e-4, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 0, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1, "betas_2": 0.98}
    configs["supernetbase.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 512, "mixing": "bert-bottleneck", "max_train_steps": 125000, "tokenized_c4_dir": dataset_factory("wikibooks_tokenized"), "model_name_or_path": "bert-base-cased", "sampling_type": "random", "sampling_rule": "sandwich", "learning_rate": 1e-4, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 0, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1, "betas_2": 0.98}
    # CUDA OOM issue
    configs["supernetbase.standard.train_mlm"] = modify_config(configs["supernetbase.standard.train_mlm"], {"per_device_train_batch_size": "16", "gradient_accumulation_steps": "16", "per_device_eval_batch_size": "16"})
    configs["supernetbase.standard.train_mlm.32bsz"] = modify_config(configs["supernetbase.standard.train_mlm"], {"per_device_train_batch_size": "32", "gradient_accumulation_steps": "8", "per_device_eval_batch_size": "32"})
    configs["supernetbase.standard.train_glue"] = {"learning_rate": 5e-05, "mixing": "bert-bottleneck", "model_name_or_path": "", "num_train_epochs": 10, "per_device_train_batch_size": 4, "sampling_type": "none", "task": "", "subtransformer_config_path": "", "output_dir": "<<OUTPUT_DIR>>"}

    # phase-3 after reading academic BERT paper https://arxiv.org/pdf/2104.07705.pdf
    configs["supernetbasev3.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 128, "mixing": "bert-bottleneck", "max_train_steps": 125000, "tokenized_c4_dir": dataset_factory("wikibooks_graphcore_128len_next_sentence_label_removed_w_splits"), "model_name_or_path": "bert-base-uncased", "sampling_type": "random", "sampling_rule": "sandwich", "learning_rate": 5e-4, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 0, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1, "betas_2": 0.98, "tokenizer_name": "Graphcore/bert-base-uncased"}
    configs["bertbasev3.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 128, "mixing": "bert-bottleneck", "max_train_steps": 125000, "tokenized_c4_dir": dataset_factory("wikibooks_graphcore_128len_next_sentence_label_removed_w_splits"), "model_name_or_path": "bert-base-uncased", "sampling_type": "none", "sampling_rule": "none", "learning_rate": 5e-4, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 0, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1, "betas_2": 0.98, "tokenizer_name": "Graphcore/bert-base-uncased"}
    configs["supernetbasev3.standard.train_glue"] = {"learning_rate": 5e-05, "mixing": "bert-bottleneck", "model_name_or_path": "", "num_train_epochs": 10, "per_device_train_batch_size": 4, "sampling_type": "none", "task_name": "", "output_dir": "<<OUTPUT_DIR>>", "tokenizer_name": "Graphcore/bert-base-uncased", "skip_saving_checkpoints": "yes"} # "subtransformer_config_path": ""
    # configs["supernetbasev3.standard.train_glue_original"] = {"learning_rate": 5e-05, "model_name_or_path": "", "num_train_epochs": 10, "per_device_train_batch_size": 4, "task_name": "", "output_dir": "<<OUTPUT_DIR>>", "tokenizer_name": "Graphcore/bert-base-uncased"} 
    # configs["supernetbasev3.standard.nockptsavings.train_glue_original"] =  modify_config(configs["supernetbasev3.standard.train_glue_original"], {"skip_saving_checkpoints" : "yes"})
    configs["supernetbasev3.evosearch"] =  {"output_dir": "<<OUTPUT_DIR>>", "mixing": "bert-bottleneck", "supernet_ckpt_dir": "", "data_dir": dataset_factory("wikibooks_graphcore_128len_next_sentence_label_removed_w_splits")}
    configs["supernetbasev3.standard.train_glue_original"] = {"learning_rate": "", "model_name_or_path": "", "num_train_epochs": 10, "per_device_train_batch_size": "", "task_name": "", "output_dir": "<<OUTPUT_DIR>>", "tokenizer_name": "Graphcore/bert-base-uncased",  "skip_saving_checkpoints": "yes", "seed": "333"} 

    return configs[name]

def modify_config_and_to_string(prev_config, updates):
    for update_key in updates:
        prev_config[update_key] = updates[update_key]
    out_str = ""
    for k in prev_config:
        out_str += "--%s %s "%(k, str(prev_config[k]))
    return out_str.strip()

def modify_config(prev_config, updates):
    for update_key in updates:
        prev_config[update_key] = updates[update_key]
    return prev_config

def get_experiments_dir():
    return "/fsx/ganayu/experiments/supershaper"

def get_glue_datasets():
    return ["mnli", "cola", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"] #, "wnli"]

def get_model_configs(model_name):
    configs = {}

    # bert 12L_120H vs. 12L_768H
    configs["bert.bottleneck.12L_120H"] = "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_120H.csv"
    configs["bert.bottleneck.12L_768H"] = "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_768H.csv"
    configs["bertuncased.bottleneck.12L_120H"] = "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_uncased_12L_120H.csv"
    configs["bertuncased.bottleneck.12L_768H"] = "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_uncased_12L_768H.csv"
    configs["bertuncased.nobottleneck.12L_768H"] = "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_uncased_nobottleneck_12L_768H.csv"

    return configs[model_name]

def get_finetuning_sweeps(method):
    sweeps = {}
    sweeps["roberta"] = {"ft_lrs":["1e-5", "2e-5", "3e-5"], "ft_bsz":["16", "32"], "ft_epochs":["10"], "warmup_ratio": 0.06}
    sweeps["wellreadstudents"] = {"ft_lrs":["3e-4", "1e-4", "5e-5", "3e-5"], "ft_bsz":["8", "16", "32", "64", "128"], "ft_epochs":["4"]}
    sweeps["supershaper"] = {"ft_lrs":["5e-5"], "ft_bsz":["32"], "ft_epochs":["10"]}
    sweeps["summer"] = {"ft_lrs": [
    "1e-6", "2e-6", "5e-6", "1e-5", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3", "2e-3", "5e-3"], "ft_bsz":["8", "16", "32"], "ft_epochs":["10"]}
    sweeps["bert"] = {"ft_lrs":["5e-5", "3e-5", "2e-5"], "ft_bsz":["16", "32"], "ft_epochs":["2", "3", "4"]}
    return sweeps[method]

'''
phase-1 slide upto 27
'''

# script_creator(get_experiments_dir() + "/jul8_std_supershaper", [ {"exp_name": "seqlen_128", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_mlm"), {"eval_random_subtransformers": 0})}]}, {"exp_name": "seqlen_512", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_mlm"), {"max_seq_length": 512, "eval_random_subtransformers": 0})}]}  ] )
# script_creator(get_experiments_dir() + "/jul9_wikibooks_inplace_distillation", [ {"exp_name": "seqlen_128", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_mlm"), {"eval_random_subtransformers": 0, "inplace_distillation": 1, "layerwise_distillation": 1, "c4_dir": dataset_factory("wikibooks")})}]}  ] )
# script_creator(get_experiments_dir() + "/jul10_wikibooks_lr5e-5", [ {"exp_name": "seqlen_128", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_mlm"), {"eval_random_subtransformers": 0, "learning_rate": 5e-5,  "c4_dir": dataset_factory("wikibooks")})}]}  ] )
# script_creator(get_experiments_dir() + "/jul10_wikibooks_supernet_train_more_steps", [ {"exp_name": "seqlen_128", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_mlm"), {"eval_random_subtransformers": 0, "max_train_steps": 350000, "resume_from_checkpoint_dir": "/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model", "c4_dir": dataset_factory("wikibooks")})}]}  ] )
# script_creator(get_experiments_dir() + "/jul11_wikibooks_finetune_9tasks", [ {"exp_name": "seqlen_128_%s"%TASK, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_glue"), {"model_name_or_path": "/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv", "task": TASK})}]}  for TASK in get_glue_datasets() ] )
# script_creator(get_experiments_dir() + "/jul11_wikibooks_efficient_subnet_train_more_steps", [ {"exp_name": "seqlen_128", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_mlm"), {"eval_random_subtransformers": 0, "max_train_steps": 50000, "sampling_type": "none", "sampling_rule": "none", "model_name_or_path": "/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv", "c4_dir": dataset_factory("wikibooks")})}]}  ] )
# script_creator(get_experiments_dir() + "/jul11_wikibooks_efficient_subnet_train_more_steps_100K", [ {"exp_name": "seqlen_128", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_mlm"), {"eval_random_subtransformers": 0, "max_train_steps": 100000, "sampling_type": "none", "sampling_rule": "none", "model_name_or_path": "/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv", "c4_dir": dataset_factory("wikibooks")})}]}  ] )
# script_creator(get_experiments_dir() + "/jul12_wikibooks_efficient_subnet_train_more_steps_finetune_9tasks", [ {"exp_name": "seqlen_128_%s"%TASK, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_glue"), {"model_name_or_path": "/fsx/ganayu/experiments/supershaper/jul11_wikibooks_efficient_subnet_train_more_steps/seqlen_128/c4_realnews_bert-bottleneck_none_K=1_pretraining_seqlen_128_13-07-2022-00-00-19/best_model", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv", "task": TASK})}]}  for TASK in get_glue_datasets() ] )
# script_creator(get_experiments_dir() + "/jul12_wikibooks_efficient_supernet_train_more_steps_finetune_9tasks", [ {"exp_name": "seqlen_128_%s"%TASK, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_glue"), {"model_name_or_path": "/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model/best_model", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv", "task": TASK})}]}  for TASK in get_glue_datasets() ] )
# script_creator(get_experiments_dir() + "/jul12_wikibooks_lr5e-5_finetune_9tasks", [ {"exp_name": "seqlen_128_%s"%TASK, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_glue"), {"model_name_or_path": "/fsx/ganayu/experiments/supershaper/jul10_wikibooks_lr5e-5/seqlen_128/c4_realnews_bert-bottleneck_random_K=1_pretraining_seqlen_128_11-07-2022-05-19-41/best_model", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv", "task": TASK})}]}  for TASK in get_glue_datasets() ] )
# script_creator(get_experiments_dir() + "/jul12_wikibooks_efficient_subnet_train_more_steps_100K_finetune_9tasks", [ {"exp_name": "seqlen_128_%s"%TASK, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supershaper.standard.train_glue"), {"model_name_or_path": "/fsx/ganayu/experiments/supershaper/jul11_wikibooks_efficient_subnet_train_more_steps_100K/seqlen_128/c4_realnews_bert-bottleneck_none_K=1_pretraining_seqlen_128_13-07-2022-00-07-28/best_model", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv", "task": TASK})}]}  for TASK in get_glue_datasets() ] )

'''
phase-2
changes:
1) BERT lr 1e-4
2) max seq len 512
3) NO line by line & Merge sentences from different docs
4) max steps is 125K
5) validation set back to 5% (from 50K)
6) roberts' AdamW change B_2=0.98 for stability with bigger batches

train independent models
'''
# supernetbase - base, base_longer250K, base_lr6e-4
# script_creator(get_experiments_dir() + "/jul16_supernetbase_initial", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm"), {"experiment_name": "jul14_supernetbase_initial_base"})}]}  ]) #, {"exp_name": "base_longer250K", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm"), {"max_train_steps": 250000, "experiment_name": "jul14_supernetbase_initial_base_longer250K"})}]}, {"exp_name": "base_lr6e-4", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm"), {"learning_rate": 6e-4, "experiment_name": "jul14_supernetbase_initial_base_lr6e-4"})}]} ] )
# script_creator(get_experiments_dir() + "/jul17_supernetbase_initial_largebsz", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm.32bsz"), {"experiment_name": "jul17_supernetbase_initial_largebsz"})}]}  ]) 

# train largest transformer bert-base 768H 12L (110M) w bottleneck
# train smallest transformer bert-base 120H 12L w bottleneck
# script_creator(get_experiments_dir() + "/jul16_bertstandalone", [ {"exp_name": "12L_768H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_768H.csv"})}]}, {"exp_name": "12L_120H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_120H.csv"})}]}  ] )
# script_creator(get_experiments_dir() + "/jul17_bertstandalone_largebsz", [ {"exp_name": "12L_768H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm.32bsz"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_768H.csv", "experiment_name": "jul17_bertstandalone_largebsz_12L_768H"})}]}, {"exp_name": "12L_120H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm.32bsz"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_120H.csv", "experiment_name": "jul17_bertstandalone_largebsz_12L_120H"})}]}  ] )

# supernet - train with sample_one_arch=True
# script_creator(get_experiments_dir() + "/jul18_supernetbase_sample1arch", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm.32bsz"), {"experiment_name": "jul18_supernetbase_sample1arch", "sampling_rule": "none", "sample_one_arch": "yes" })}]}  ]) 

# finetuning on mnli - standalone vs supernet
def create_finetuning_experiments_standalone_vs_supernet(epochs_to_consider, supernet_master_dir, standalone_master_dir, configs=["bert.bottleneck.12L_120H", "bert.bottleneck.12L_768H"], tasks=["mnli"]):
    experiments = []
    for task in tasks:
        for epoch in epochs_to_consider:
            # supernet
            if supernet_master_dir:
                supernet_ckpt = get_experiments_dir() + "/%s/base/epoch_%d"%(supernet_master_dir, epoch)
                for model_name in configs:
                    config = get_model_configs(model_name)
                    experiment_name = "supernet_%s_ep%d_%s"%(task, epoch, model_name)
                    experiments.append({"exp_name": experiment_name, "experiment_name": experiment_name, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_glue"), {"model_name_or_path": supernet_ckpt, "subtransformer_config_path": config, "task": task })}]})
            # standalone
            if standalone_master_dir:
                for model_name in configs:
                    config = get_model_configs(model_name)
                    effnet_ckpt =  get_experiments_dir() + "/%s/%s/epoch_%d"%(standalone_master_dir, model_name.split(".")[-1], epoch)
                    experiment_name = "standalone_%s_ep%d_%s"%(task, epoch, model_name)
                    experiments.append({"exp_name": experiment_name, "experiment_name": experiment_name, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_glue"), {"model_name_or_path": effnet_ckpt, "subtransformer_config_path": config, "task": task})}]})
    return experiments
experiments = []
epochs_to_consider = [2]
for epoch in epochs_to_consider:
    # supernet
    supernet_ckpt = get_experiments_dir() + "/jul17_supernetbase_initial_largebsz/base/epoch_%d"%epoch
    for model_name, config in [ ("bert.bottleneck.12L_120H", get_model_configs("bert.bottleneck.12L_120H")), ("bert.bottleneck.12L_768H", get_model_configs("bert.bottleneck.12L_768H"))]:
        experiment_name = "supernet_mnli_ep%d_%s"%(epoch, model_name)
        experiments.append({"exp_name": experiment_name, "experiment_name": experiment_name, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_glue"), {"model_name_or_path": supernet_ckpt, "subtransformer_config_path": config, "task": "mnli"})}]})
    # standalone
    for model_name, config in [ ("bert.bottleneck.12L_120H", get_model_configs("bert.bottleneck.12L_120H")), ("bert.bottleneck.12L_768H", get_model_configs("bert.bottleneck.12L_768H"))]:
        effnet_ckpt =  get_experiments_dir() + "/jul17_bertstandalone_largebsz/%s/epoch_%d"%(model_name.split(".")[-1], epoch)
        experiment_name = "standalone_mnli_ep%d_%s"%(epoch, model_name)
        experiments.append({"exp_name": experiment_name, "experiment_name": experiment_name, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_glue"), {"model_name_or_path": effnet_ckpt, "subtransformer_config_path": config, "task": "mnli"})}]})
# experiments = create_finetuning_experiments_standalone_vs_supernet(epochs_to_consider=[2], supernet_master_dir="jul17_supernetbase_initial_largebsz", standalone_master_dir="jul17_bertstandalone_largebsz")
# script_creator(get_experiments_dir() + "/jul18_supernetbase_vs_standalone_finetune_mnli", experiments, time_in_mins=600, wandb="offline")

'''
phase-3
bert-base uncased
128 seq len
lr = 5e-4 (supershaper)
dataset - graphcore

todo:
mixing attention for standard bert model
turn off pretrained weights initialization
'''
# script_creator(get_experiments_dir() + "/jul19_v3_bertstandalone", [ {"exp_name": "12L_768H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_uncased_12L_768H.csv", "experiment_name": "jul19_v3_bertstandalone_12L_768H"})}]}, {"exp_name": "12L_120H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_uncased_12L_120H.csv", "experiment_name": "jul19_v3_bertstandalone_12L_120H"})}]}  ], time_in_mins=10000)
# script_creator(get_experiments_dir() + "/jul19_v3_supernetbase", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul19_v3_supernetbase"})}]}  ], time_in_mins=10000) 
# experiments = create_finetuning_experiments_standalone_vs_supernet([7], None, "/fsx/ganayu/experiments/supershaper/jul19_v3_bertstandalone", configs=["bertuncased.bottleneck.12L_120H", "bertuncased.bottleneck.12L_768H"], tasks=["mrpc"])
# script_creator(get_experiments_dir() + "/jul20_v3_standalone_finetune_mrpc", experiments, time_in_mins=120, wandb="offline")

# bert base 110M - standalone - no preinit - no bottleneck 
# script_creator(get_experiments_dir() + "/jul21_v3_bertstandalone_noinit_nobottleneck", [ {"exp_name": "12L_768H_5e-4", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": get_model_configs("bertuncased.nobottleneck.12L_768H"), "initialize_pretrained_weights": "no", "mixing": "attention", "experiment_name": "jul21_v3_bertstandalone_noinit_nobottleneck_12L_768H_5e-4"})}]},  {"exp_name": "12L_768H_1e-3", "runs": [{"pyfile": "train_mlm.py", "params": # modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": get_model_configs("bertuncased.nobottleneck.12L_768H"), "initialize_pretrained_weights": "no", "mixing": "attention", "learning_rate": 1e-3, "experiment_name": "jul21_v3_bertstandalone_noinit_nobottleneck12L_768H_1e-3"})}]}  ], time_in_mins=5000, wandb="online")

# compute test PPL for baselines and ours.
# script_creator(get_experiments_dir() + "/jul21_v3_test_ppl_baselines", [ {"exp_name": exp_name, "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": get_model_configs(model_config), "mixing": mixing, "experiment_name": "jul21_v3_%s"%exp_name, "tokenizer_name": tokenizer_name, "model_name_or_path": model_name_or_path, "check_test_loss_only": "yes"})}]} for exp_name, model_name_or_path, tokenizer_name, mixing, model_config in [("google-bert", "bert-base-uncased", "bert-base-uncased", "attention", "bertuncased.nobottleneck.12L_768H"), ("graphcore-bert", "Graphcore/bert-base-uncased", "Graphcore/bert-base-uncased", "attention", "bertuncased.nobottleneck.12L_768H"), ("ourbert-w-initbottle", "/fsx/ganayu/experiments/supershaper/jul19_v3_bertstandalone/12L_768H/best_model", "Graphcore/bert-base-uncased", "bert-bottleneck", "bertuncased.bottleneck.12L_768H")]], time_in_mins=300, wandb="offline", num_gpus=1)
# compute valid MNLI score for baseline and ours following RoBERTa sweep
def create_finetuning_experiments_standalone_vs_supernet_v2(models=[ ("graphcore-bert", "Graphcore/bert-base-uncased", "Graphcore/bert-base-uncased", "attention", "bertuncased.nobottleneck.12L_768H"), ("ourbert-w-initbottle", "/fsx/ganayu/experiments/supershaper/jul19_v3_bertstandalone/12L_768H/best_model", "Graphcore/bert-base-uncased", "bert-bottleneck", "bertuncased.bottleneck.12L_768H"), ("google-bert", "bert-base-uncased", "bert-base-uncased", "attention", "bertuncased.nobottleneck.12L_768H")], configs=["bert.bottleneck.12L_120H", "bert.bottleneck.12L_768H"], tasks=["mnli"], sweep={"ft_lrs":["1e-5", "2e-5", "3e-5"], "ft_bsz":["16", "32"], "ft_epochs":["10"]}, glue_config="supernetbasev3.standard.train_glue", num_gpus=1):
    experiments = []
    for task in tasks:
        for lr in sweep["ft_lrs"]:
            for bsz in sweep["ft_bsz"]:
                for epoch in sweep["ft_epochs"]:
                    for exp_name, model_name_or_path, tokenizer_name, mixing, model_config in models:
                        cur_exp_name = "%d_%s_%s_%s_%s_%s"%(len(experiments), exp_name, task, lr, bsz, epoch)
                        run_config = None
                        if glue_config.endswith("train_glue"):
                            run_config = {"exp_name": cur_exp_name, "experiment_name": cur_exp_name, "runs": [{"pyfile": "train_glue.py", "params": modify_config_and_to_string(config_factory(glue_config), {"model_name_or_path": model_name_or_path,  "task_name": task, "tokenizer_name":tokenizer_name, "mixing": mixing, "learning_rate": lr, "per_device_train_batch_size": int(bsz) // num_gpus, "num_train_epochs": epoch})}]}
                            if model_config != "none":
                                run_config["subtransformer_config_path"] = get_model_configs(model_config)
                        elif glue_config.endswith("train_glue_original"):
                            if model_config == "none":
                                run_config = {"exp_name": cur_exp_name, "experiment_name": cur_exp_name, "runs": [{"pyfile": "train_glue_original.py", "params": modify_config_and_to_string(config_factory(glue_config), {"model_name_or_path": model_name_or_path, "task_name": task, "tokenizer_name": tokenizer_name, "learning_rate": lr, "per_device_train_batch_size": int(bsz) // num_gpus, "num_train_epochs": epoch})}]}
                            else:
                                run_config = {"exp_name": cur_exp_name, "experiment_name": cur_exp_name, "runs": [{"pyfile": "train_glue_original.py", "params": modify_config_and_to_string(config_factory(glue_config), {"model_name_or_path": model_name_or_path, "task_name": task, "tokenizer_name": tokenizer_name, "learning_rate": lr, "subtransformer_config_path": model_config, "per_device_train_batch_size": int(bsz) // num_gpus, "num_train_epochs": epoch})}]}
                        assert(run_config!=None)
                        experiments.append(run_config)
    return experiments

# script_creator(get_experiments_dir() + "/jul21_v3_finetune_mnli_graphcore_vs_ourbert-w-initbottle", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=1), time_in_mins=1000, wandb="offline", num_gpus=1, generate_port=True)
# script_creator(get_experiments_dir() + "/jul21_v3_8gpu1setting_finetune_mnli_graphcore_vs_ourbert-w-initbottle", create_finetuning_experiments_standalone_vs_supernet_v2(sweep={"ft_lrs":["5e-5"], "ft_bsz":["32"], "ft_epochs":["10"]}, num_gpus=8), time_in_mins=1000, wandb="offline", num_gpus=8)
#script_creator(get_experiments_dir() + "/jul22_v3_finetune_mnli_ourbert-no-initbottle", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=1, models=[ ("ourbert-no-initbottle", "/fsx/ganayu/experiments/supershaper/jul21_v3_bertstandalone_noinit_nobottleneck/12L_768H_5e-4/best_model", "Graphcore/bert-base-uncased", "attention", "bertuncased.nobottleneck.12L_768H") ]), time_in_mins=1000, wandb="offline", num_gpus=1, generate_port=True)
# script_creator(get_experiments_dir() + "/jul22_v3_finetune_mnli_oursupernet_base", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=1, models=[ ("supernetbase-12L_768H", "/fsx/ganayu/experiments/supershaper/jul19_v3_supernetbase/base/best_model", "Graphcore/bert-base-uncased", "bert-bottleneck", "bertuncased.bottleneck.12L_768H") ]), time_in_mins=1000, wandb="offline", num_gpus=1, generate_port=True)
# script_creator(get_experiments_dir() + "/jul23_v3_finetunecorrected_mnli_allmodels_slide36", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=1, tasks=["mnli"], sweep=get_finetuning_sweeps("wellreadstudents"), models=[ ("google-bert", "bert-base-uncased", "bert-base-uncased", "attention", "none"), ("graphcore-bert", "Graphcore/bert-base-uncased", "Graphcore/bert-base-uncased", "attention", "none"), ("ourbert-w-initbottle", "/fsx/ganayu/experiments/supershaper/jul19_v3_bertstandalone/12L_768H/best_model", "Graphcore/bert-base-uncased", "bert-bottleneck", "none"), ("supernetbase-12L_768H", "/fsx/ganayu/experiments/supershaper/jul19_v3_supernetbase/base/best_model", "Graphcore/bert-base-uncased", "bert-bottleneck", "none"), ("ourbert-no-initbottle", "/fsx/ganayu/experiments/supershaper/jul21_v3_bertstandalone_noinit_nobottleneck/12L_768H_5e-4/best_model", "Graphcore/bert-base-uncased", "attention", "none")]), time_in_mins=1000, wandb="offline", num_gpus=1, generate_port=True, part_size=25)
# script_creator(get_experiments_dir() + "/jul23_v3_finetunecorrected_mnli_allmodels_slide36_supershapersweep", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=1, tasks=["mnli"], sweep=get_finetuning_sweeps("supershaper"), models=[ ("google-bert", "bert-base-uncased", "bert-base-uncased", "attention", "none"), ("graphcore-bert", "Graphcore/bert-base-uncased", "Graphcore/bert-base-uncased", "attention", "none"), ("ourbert-w-initbottle", "/fsx/ganayu/experiments/supershaper/jul19_v3_bertstandalone/12L_768H/best_model", "Graphcore/bert-base-uncased", "bert-bottleneck", "none"), ("supernetbase-12L_768H", "/fsx/ganayu/experiments/supershaper/jul19_v3_supernetbase/base/best_model", "Graphcore/bert-base-uncased", "bert-bottleneck", "none"), ("ourbert-no-initbottle", "/fsx/ganayu/experiments/supershaper/jul21_v3_bertstandalone_noinit_nobottleneck/12L_768H_5e-4/best_model", "Graphcore/bert-base-uncased", "attention", "none")]), time_in_mins=1000, wandb="offline", num_gpus=1, generate_port=True, part_size=25)

'''
# bringing back "token_type_ids" and using "train_glue_original"
'''
# script_creator(get_experiments_dir() + "/jul23_v3_bertstandalone_noinit_nobottleneck_toktypecorrected", [ {"exp_name": "12L_768H_5e-4", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": get_model_configs("bertuncased.nobottleneck.12L_768H"), "initialize_pretrained_weights": "no", "mixing": "attention", "experiment_name": "jul23_v3_bertstandalone_noinit_nobottleneck_12L_768H_5e-4_toktypecorrected"})}]}], time_in_mins=5000, wandb="online")
# script_creator(get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul23_v3_supernetbase_toktypecorrected"})}]}  ], time_in_mins=10000, wandb="online") 
# script_creator(get_experiments_dir() + "/jul23_v3_bertstandalone_winit_wbottleneck_toktypecorrected", [ {"exp_name": "12L_768H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_uncased_12L_768H.csv", "experiment_name": "jul23_v3_bertstandalone_winit_wbottleneck_12L_768H_toktypecorrected"})}]}, {"exp_name": "12L_120H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_uncased_12L_120H.csv", "experiment_name": "jul23_v3_bertstandalone__winit_wbottleneck_12L_120H_toktypecorrected"})}]}  ], time_in_mins=10000)
# script_creator(get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected_inplacekd_logits_only", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul23_v3_supernetbase_toktypecorrected_inplacekd_logits_only", "inplace_distillation": 1})}]}  ], time_in_mins=10000, wandb="online") 
# script_creator(get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected_inplacekd_logits_last_attn_hid_layer", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul23_v3_supernetbase_toktypecorrected_inplacekd_logits_last_attn_hid_layer", "inplace_distillation": 1, "distillation_type": "hiddenlastlayer+attentionlastlayer"})}]}  ], time_in_mins=10000, wandb="online") 
# script_creator(get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected_2xtrainbudget", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul23_v3_supernetbase_toktypecorrected_2xtrainbudget", "max_train_steps": 250000})}]} ], time_in_mins=10000, wandb="online") 
# script_creator(get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected_noinit", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul23_v3_supernetbase_toktypecorrected_noinit", "initialize_pretrained_weights": "no"})}]} ], time_in_mins=8000, wandb="online") 


# fine-tuning - reproduce standalone baseline? can we?
# script_creator(get_experiments_dir() + "/jul24_v3_finetune_glueoriginal_mnli_toktypecorrected", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=1, tasks=["mnli"], glue_config="supernetbasev3.standard.train_glue_original", sweep=get_finetuning_sweeps("wellreadstudents"), models=[ ("google-bert", "bert-base-uncased", "bert-base-uncased", "attention", "none"), ("graphcore-bert", "Graphcore/bert-base-uncased", "Graphcore/bert-base-uncased", "attention", "none"), ("ourbert-no-initbottle", "/fsx/ganayu/experiments/supershaper/jul23_v3_bertstandalone_noinit_nobottleneck_toktypecorrected/12L_768H_5e-4/best_model", "Graphcore/bert-base-uncased", "attention", "none")]), time_in_mins=1000, wandb="no", num_gpus=1, generate_port=True, part_size=25)
# script_creator(get_experiments_dir() + "/jul24_v3_finetune_glueoriginal_sst2_cola_toktypecorrected", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=1, tasks=["sst2", "cola"], glue_config="supernetbasev3.standard.nockptsavings.train_glue_original", sweep=get_finetuning_sweeps("wellreadstudents"), models=[ ("google-bert", "bert-base-uncased", "bert-base-uncased", "attention", "none"), ("graphcore-bert", "Graphcore/bert-base-uncased", "Graphcore/bert-base-uncased", "attention", "none"), ("ourbert-no-initbottle", "/fsx/ganayu/experiments/supershaper/jul23_v3_bertstandalone_noinit_nobottleneck_toktypecorrected/12L_768H_5e-4/best_model", "Graphcore/bert-base-uncased", "attention", "none")]), time_in_mins=1000, wandb="no", num_gpus=1, generate_port=True, part_size=25)

# changing search space
# script_creator(get_experiments_dir() + "/jul24_v3_supernetbase_toktypecorrected_extending_search_space", [ {"exp_name": "hidden_layer_elastic", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul24_v3_supernetbase_toktypecorrected_hidden_layer_elastic", "search_space_id": "hidden_layer_elastic"})}]}, {"exp_name": "ffn_intermediate_elastic", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul24_v3_supernetbase_toktypecorrected_ffn_intermediate_elastic", "search_space_id": "ffn_intermediate_elastic"})}]} ], time_in_mins=8000, wandb="online")

# 2 samples during sandwich rule
# script_creator(get_experiments_dir() + "/jul25_v3_supernetbase_sandwch_2random", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul23_v3_supernetbase_toktypecorrected", "pop_size": 2})}]} ], time_in_mins=10000, wandb="online") 

# inplace kd loss
# script_creator(get_experiments_dir() + "/jul27_v3_supernetbase_inplacekd_logitshard_logitshidden", [ {"exp_name": "logits", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul27_v3_supernetbase_toktypecorrected_inplacekd_logitshard", "inplace_distillation": 1, "distillation_type": "logits+hard"})}]}, {"exp_name": "logitshidden", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul27_v3_supernetbase_toktypecorrected_inplacekd_logits_hiddenlastlayer", "inplace_distillation": 1, "distillation_type": "logits+hiddenlastlayer"})}]}  ], time_in_mins=10000, wandb="online")

# hypernet-rank
# script_creator(get_experiments_dir() + "/jul27_v3_supernetbase_hypernet", [ {"exp_name": "rank%s_hyphid%s"%(rank, hyphid), "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul27_v3_supernetbase_hypernet_rank%s_hyphid%s"%(rank, hyphid), "use_hypernet_w_low_rank": 1, "bottleneck_rank": rank, "hypernet_hidden_size": hyphid })}]}  for rank, hyphid in [("64", "50"), ("64", "16"), ("32", "16")] ], time_in_mins=10000, wandb="online")

# search space -> FFN expansion ratio
# search space -> v2 -> Intermediate expansion ratio - [2, 3, 4]; Number of layers - [6, 9, 12]; Hidden size - [120, 360, 540, 768]
# search space ->  hidden_layer_elastic corrected with val. smallest equal to v2
# script_creator(get_experiments_dir() + "/jul27_v3_supernetbase_search_space_ffn_v2_hidlayer", [ {"exp_name": search_space_id, "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul27_v3_supernetbase_search_space_%s"%search_space_id, "search_space_id": search_space_id})}]}  for search_space_id in ["hidden_layer_elastic", "ffn_intermediate_ratio_elastic", "v2"] ], time_in_mins=10000, wandb="online")

# inplace-kd -> mult distill loss by 1.0, 1.25 and 1.5
# script_creator(get_experiments_dir() + "/jul27_v3_supernetbase_inplacekd_distill_contrib", [ {"exp_name": inplace_kd_distill_loss_weights, "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul27_v3_supernetbase_inplacekd_distill_contrib_%s"%inplace_kd_distill_loss_weights, "inplace_distillation": 1, "distillation_type": "logits", "inplace_kd_distill_loss_weights": inplace_kd_distill_loss_weights})}]} for inplace_kd_distill_loss_weights in ["1.0", "1.5", "2"] ], time_in_mins=10000, wandb="online")

# playing with batch size
# script_creator(get_experiments_dir() + "/jul30_v3_supernetbase_nasbert_bsz_1024_250Ksteps", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "jul30_v3_supernetbase_nasbert_bsz_1024_250Ksteps", "max_train_steps": 250000, "gradient_accumulation_steps": 1})}]}  ], time_in_mins=10000, wandb="online")

# search - different spaces
# script_creator(get_experiments_dir() + "/aug1_v3_supernetbase_search_different_spaces", [ {"exp_name": "hiddenonly", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected/base/best_model"})}]},   {"exp_name": "hidden_layer_elastic", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul27_v3_supernetbase_search_space_ffn_v2_hidlayer/hidden_layer_elastic/best_model", "search_space_id": "hidden_layer_elastic"})}]},    {"exp_name": "ffn_intermediate_ratio_elastic", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul27_v3_supernetbase_search_space_ffn_v2_hidlayer/ffn_intermediate_ratio_elastic/best_model", "search_space_id": "ffn_intermediate_ratio_elastic"})}]},  {"exp_name": "v2", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul27_v3_supernetbase_search_space_ffn_v2_hidlayer/v2/best_model", "search_space_id": "v2"})}]},    ], time_in_mins=8000, wandb="no")

# script_creator(get_experiments_dir() + "/aug1_v3_supernetbase_search_diff_configs", [ {"exp_name": "2xtrainbudget", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected_2xtrainbudget/base/best_model"})}]}, {"exp_name": "noinit", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected_noinit/base/best_model"})}]},  {"exp_name": "sandwch_2random", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul25_v3_supernetbase_sandwch_2random/base/best_model"})}]}   ], time_in_mins=8000, wandb="no")

# inplace search
# script_creator(get_experiments_dir() + "/aug1_v3_supernetbase_search_inplace_configs", [ {"exp_name": "logitshard", "runs": [{"pyfile": "evo_ours.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.evosearch"), {"supernet_ckpt_dir": get_experiments_dir() + "/jul27_v3_supernetbase_inplacekd_logitshard_logitshidden/logits/best_model"})}]}  ], time_in_mins=8000, wandb="no")

# train for more steps
# script_creator(get_experiments_dir() + "/aug1_v3_evosearch_hiddenlayeronly_subnet_train_more_steps", [ {"exp_name": "stage1", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "aug1_v3_evosearch_hiddenlayeronly_subnet_train_more_steps_stage1", "max_train_steps": 100000, "sampling_type": "none", "sampling_rule": "none", "model_name_or_path": get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected/base/best_model", "subtransformer_config_path": get_experiments_dir() + "/aug1_v3_supernetbase_search_different_spaces/hiddenonly/evo_results_29.xls" })}]},  {"exp_name": "nowarmup", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "aug1_v3_evosearch_hiddenlayeronly_subnet_train_more_steps_nowarmup", "max_train_steps": 100000, "num_warmup_steps": 0, "sampling_type": "none", "sampling_rule": "none", "model_name_or_path": get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected/base/best_model", "subtransformer_config_path": get_experiments_dir() + "/aug1_v3_supernetbase_search_different_spaces/hiddenonly/evo_results_29.xls" })}]},   {"exp_name": "5timeslowerlr", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "aug1_v3_evosearch_hiddenlayeronly_subnet_train_more_steps_5timeslowerlr", "max_train_steps": 100000, "learning_rate": 1e-05, "sampling_type": "none", "sampling_rule": "none", "model_name_or_path": get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected/base/best_model", "subtransformer_config_path": get_experiments_dir() + "/aug1_v3_supernetbase_search_different_spaces/hiddenonly/evo_results_29.xls" })}]}  ], time_in_mins=8000, wandb="online") 

# direct finetuning
def create_finetuning_experiments_standalone_vs_supernet_v3(models=None, tasks=["mnli"], sweep=None, glue_config=None, num_gpus=4):
    experiments = []
    for exp_name, model_name_or_path, subtransformer_config_path, tokenizer_name in models:
        run_config = {"exp_name": exp_name, "runs": []}
        for task in tasks:
            for lr in sweep["ft_lrs"]:
                for bsz in sweep["ft_bsz"]:
                    for epoch in sweep["ft_epochs"]:
                        updated_params_config = {"model_name_or_path": model_name_or_path, "task_name": task, "tokenizer_name": tokenizer_name, "learning_rate": lr, "per_device_train_batch_size": int(bsz) // num_gpus, "num_train_epochs": epoch}
                        if subtransformer_config_path:
                            updated_params_config["subtransformer_config_path"] = subtransformer_config_path
                        for param in sweep:
                            if not param.startswith("ft"):
                                updated_params_config[param] = sweep[param]
                        run_info = {"pyfile": "train_glue_original.py", "params": modify_config_and_to_string(config_factory(glue_config), updated_params_config) }
                        run_config["runs"].append(run_info)
        experiments.append(run_config)
    return experiments
models = []
# exp_name, model_name_or_path, subtransformer_config_path, tokenizer_name
models.append(("supershaper", "/fsx/ganayu/experiments/supershaper/jul23_v3_supernetbase_toktypecorrected/base/best_model", "/fsx/ganayu/experiments/supershaper/aug1_v3_supernetbase_search_different_spaces/hiddenonly/evo_results_29.xls", "Graphcore/bert-base-uncased"))
models.append(("2xtrainbudget", "/fsx/ganayu/experiments/supershaper/jul23_v3_supernetbase_toktypecorrected_2xtrainbudget/base/best_model", "/fsx/ganayu/experiments/supershaper/aug1_v3_supernetbase_search_diff_configs/2xtrainbudget/evo_results_29.xls", "Graphcore/bert-base-uncased"))
models.append(("sandwch_2random", "/fsx/ganayu/experiments/supershaper/jul25_v3_supernetbase_sandwch_2random/base/best_model", "/fsx/ganayu/experiments/supershaper/aug1_v3_supernetbase_search_diff_configs/sandwch_2random/evo_results_29.xls", "Graphcore/bert-base-uncased"))
# for sweep in ["roberta", "bert", "wellreadstudents", "supershaper", "summer"]:
#    experiments = create_finetuning_experiments_standalone_vs_supernet_v3(models=[models[0]], sweep=get_finetuning_sweeps(sweep), glue_config="supernetbasev3.standard.train_glue_original", tasks=["mrpc", "cola"], num_gpus=4)
#    script_creator(get_experiments_dir() + "/aug2_supershaper_directfinetune_sweepcheck_%s_mrpc_cola"%(sweep), experiments, time_in_mins=10000, wandb="no", num_gpus=4, generate_port=True, part_size=25)
# experiments = create_finetuning_experiments_standalone_vs_supernet_v3(models=models, sweep=get_finetuning_sweeps("bert"), glue_config="supernetbasev3.standard.train_glue_original", tasks=["mnli", "cola", "mrpc", "sst2"], num_gpus=8)
# script_creator(get_experiments_dir() + "/aug2_directfinetune_mnli_supershaper_2xtrainbudget_sandwch_2random", experiments, time_in_mins=10000, wandb="no", num_gpus=8, generate_port=True, part_size=25)

# directfinetuning - one
# script_creator(get_experiments_dir() + "/aug2_v3_finetune_supershaper_60M_direct", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=4, tasks=["mnli"], glue_config="supernetbasev3.standard.train_glue_original", sweep=get_finetuning_sweeps("bert"), models=[ ("supershaper-60M-direct", "/fsx/ganayu/experiments/supershaper/jul23_v3_supernetbase_toktypecorrected/base/best_model", "Graphcore/bert-base-uncased", None, "/fsx/ganayu/experiments/supershaper/aug1_v3_supernetbase_search_different_spaces/hiddenonly/evo_results_29.xls")]), time_in_mins=1000, wandb="no", num_gpus=4, generate_port=True, part_size=5)
# script_creator(get_experiments_dir() + "/aug2_v3_finetune_v2_60M_direct", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=4, tasks=["mnli"], glue_config="supernetbasev3.standard.train_glue_original", sweep=get_finetuning_sweeps("bert"), models=[ ("v2-60M-direct", get_experiments_dir() + "/jul27_v3_supernetbase_search_space_ffn_v2_hidlayer/v2/best_model", "Graphcore/bert-base-uncased", None, get_experiments_dir() + "/aug1_v3_supernetbase_search_different_spaces/v2/evo_results_29.xls")]), time_in_mins=1000, wandb="no", num_gpus=4, generate_port=True, part_size=5)
# 100Kfinetuning - one
# script_creator(get_experiments_dir() + "/aug2_v3_finetune_supershaper_60M_100Ksteps", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=4, tasks=["mnli"], glue_config="supernetbasev3.standard.train_glue_original", sweep=get_finetuning_sweeps("bert"), models=[ ("supershaper-60M-100Ksteps", "/fsx/ganayu/experiments/supershaper/aug1_v3_evosearch_hiddenlayeronly_subnet_train_more_steps/nowarmup/best_model", "Graphcore/bert-base-uncased", None, "/fsx/ganayu/experiments/supershaper/aug1_v3_supernetbase_search_different_spaces/hiddenonly/evo_results_29.xls")]), time_in_mins=1000, wandb="no", num_gpus=4, generate_port=True, part_size=5)

# continue pretraining - 2xtrainbudget, v2
# script_creator(get_experiments_dir() + "/aug1_v3_evosearch_v2_2xbudget_subnet_train_more_steps", [ {"exp_name": "v2", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "aug1_v3_evosearch_v2_2xbudget_subnet_train_more_steps_v2", "max_train_steps": 100000, "sampling_type": "none", "num_warmup_steps": 0, "sampling_rule": "none", "model_name_or_path": get_experiments_dir() + "/jul27_v3_supernetbase_search_space_ffn_v2_hidlayer/v2/best_model", "subtransformer_config_path": get_experiments_dir() + "/aug1_v3_supernetbase_search_different_spaces/v2/evo_results_29.xls" })}]},  {"exp_name": "2xbudget", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "aug1_v3_evosearch_hiddenlayeronly_subnet_train_more_steps_2xbudget", "max_train_steps": 100000, "num_warmup_steps": 0, "sampling_type": "none", "sampling_rule": "none", "model_name_or_path": get_experiments_dir() + "/jul23_v3_supernetbase_toktypecorrected_2xtrainbudget/base/best_model", "subtransformer_config_path": get_experiments_dir() + "/aug1_v3_supernetbase_search_diff_configs/2xtrainbudget/evo_results_29.xls" })}]}  ], time_in_mins=8000, wandb="online") 
# finetune the above two
# script_creator(get_experiments_dir() + "/aug4_v3_finetune_v2_2xbudget_60M_100Ksteps", create_finetuning_experiments_standalone_vs_supernet_v2(num_gpus=4, tasks=["mnli"], glue_config="supernetbasev3.standard.train_glue_original", sweep=get_finetuning_sweeps("bert"), models=[ ("v2", get_experiments_dir() + "/aug1_v3_evosearch_v2_2xbudget_subnet_train_more_steps/v2/best_model", "Graphcore/bert-base-uncased", None, get_experiments_dir() + "/aug1_v3_supernetbase_search_different_spaces/v2/evo_results_29.xls"), ("2xbudget", get_experiments_dir() + "/aug1_v3_evosearch_v2_2xbudget_subnet_train_more_steps/2xbudget/best_model", "Graphcore/bert-base-uncased", None, get_experiments_dir() + "/aug1_v3_supernetbase_search_diff_configs/2xtrainbudget/evo_results_29.xls")]), time_in_mins=1000, wandb="no", num_gpus=4, generate_port=True, part_size=5)

# continue pretraining - sandwich2rand
# script_creator(get_experiments_dir() + "/aug4_v3_sandwch_2rand_subnet_train_more_steps", [ {"exp_name": "sandwch_2rand", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbasev3.standard.train_mlm"), {"experiment_name": "aug4_v3_sandwch_2rand_subnet_train_more_steps_sandwch_2rand", "max_train_steps": 100000, "sampling_type": "none", "num_warmup_steps": 0, "sampling_rule": "none", "model_name_or_path": get_experiments_dir() + "/jul25_v3_supernetbase_sandwch_2random/base/best_model", "subtransformer_config_path": get_experiments_dir() + "/aug1_v3_supernetbase_search_diff_configs/sandwch_2random/evo_results_29.xls" })}]}], time_in_mins=8000, wandb="online")

