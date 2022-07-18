'''
utilities to start jobs in AWS
'''

import sys, os, json, glob, shutil

def script_creator(output_folder, commands, time_in_mins=15000, num_gpus=8, code_dir="/fsx/ganayu/code/SuperShaper", wandb="online", wandb_entity="ganayu", wandb_project="effbert"):
    print("ensure you are in login node and 'basic' conda environment")
    # create output folder
    os.makedirs(output_folder, exist_ok=True)
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
    master_shell_f = output_folder + "/master.sh"
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
        w_sbatch.write("srun --label %s"%cur_shell_f)
        w_sbatch.close()
        w_shell = open(cur_shell_f, "w")
        w_shell.write("#!/bin/bash\n")
        w_shell.write("wandb %s\n"%(wandb))
        w_shell.write("TOKENIZERS_PARALLELISM=false\n")
        w_shell.write("cd %s\n"%cur_code_dir)
        for run in command["runs"]:
            if "<<OUTPUT_DIR>>" in run["params"]:
                run["params"] = run["params"].replace("<<OUTPUT_DIR>>", cur_out_dir)
            w_shell.write("accelerate launch --config_file %s %s %s"%(cur_envscript, run["pyfile"], run["params"]))
            w_shell.write(" --wandb_suffix %s --wandb_entity %s --wandb_project %s"%(command["exp_name"], wandb_entity, wandb_project))
            w_shell.write("\n")
        w_shell.close()
        w_master.write("sbatch %s\n"%(cur_sbatch_f))
        os.system("chmod 777 %s/*.sh"%(cur_out_dir))
    w_master.close()
    os.system("chmod 777 %s/*.sh"%(output_folder))
    print('bash %s'%(master_shell_f))

def dataset_factory(name):
    datasets = {}
    
    # bert-pretraining data
    datasets["wikibooks"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final"
    datasets["wikibooks_dummy"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy"
    datasets["wikibooks_tokenized"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final_tokenized"
    datasets["wikibooks_dummy_tokenized"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy_tokenized"

    # c4-pretraining data
    datasets["c4"] = "/fsx/ganayu/data/supershaper_pretraining_data/c4_datasets_cleaned"

    return datasets[name]

def config_factory(name):
    configs = {}
    # std. supershaper
    configs["supershaper.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 128, "mixing": "bert-bottleneck", "max_train_steps": 175214, "c4_dir": dataset_factory("c4"), "model_name_or_path": "bert-base-cased", "sampling_type": "random", "sampling_rule": "sandwich", "learning_rate": 2e-5, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 1, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1}
    configs["supershaper.standard.train_glue"] = {"learning_rate": 5e-05, "mixing": "bert-bottleneck", "model_name_or_path": "", "num_train_epochs": 10, "per_device_train_batch_size": 4, "sampling_type": "none", "task": "", "subtransformer_config_path": "", "output_dir": "<<OUTPUT_DIR>>"}

    # phase-2 changes
    # 125000*2048 == 1000000*256
    configs["bertbase.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 512, "mixing": "attention", "max_train_steps": 125000, "c4_dir": dataset_factory("wikibooks"), "model_name_or_path": "bert-base-cased", "sampling_type": "none", "sampling_rule": "none", "learning_rate": 1e-4, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 0, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1, "betas_2": 0.98}
    configs["supernetbase.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 512, "mixing": "bert-bottleneck", "max_train_steps": 125000, "tokenized_c4_dir": dataset_factory("wikibooks_tokenized"), "model_name_or_path": "bert-base-cased", "sampling_type": "random", "sampling_rule": "sandwich", "learning_rate": 1e-4, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 0, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1, "betas_2": 0.98}
    # CUDA OOM issue
    configs["supernetbase.standard.train_mlm"] = modify_config(configs["supernetbase.standard.train_mlm"], {"per_device_train_batch_size": "16", "gradient_accumulation_steps": "16", "per_device_eval_batch_size": "16"})
    configs["supernetbase.standard.train_mlm.32bsz"] = modify_config(configs["supernetbase.standard.train_mlm"], {"per_device_train_batch_size": "32", "gradient_accumulation_steps": "8", "per_device_eval_batch_size": "32"})

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
script_creator(get_experiments_dir() + "/jul17_supernetbase_initial_largebsz", [ {"exp_name": "base", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm.32bsz"), {"experiment_name": "jul17_supernetbase_initial_largebsz"})}]}  ]) 

# train largest transformer bert-base 768H 12L (110M) w bottleneck
# train smallest transformer bert-base 120H 12L w bottleneck
# script_creator(get_experiments_dir() + "/jul16_bertstandalone", [ {"exp_name": "12L_768H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_768H.csv"})}]}, {"exp_name": "12L_120H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_120H.csv"})}]}  ] )
script_creator(get_experiments_dir() + "/jul17_bertstandalone_largebsz", [ {"exp_name": "12L_768H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm.32bsz"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_768H.csv", "experiment_name": "jul17_bertstandalone_largebsz_12L_768H"})}]}, {"exp_name": "12L_120H", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("supernetbase.standard.train_mlm.32bsz"), {"sampling_type": "none", "sampling_rule": "none", "subtransformer_config_path": "/fsx/ganayu/experiments/supershaper/configs/bert/bertbase_12L_120H.csv", "experiment_name": "jul17_bertstandalone_largebsz_12L_120H"})}]}  ] )



