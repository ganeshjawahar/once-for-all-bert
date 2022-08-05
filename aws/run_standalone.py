'''
scripts to run standalone BERT-pretraining (no bottleneck) in AWS

todo: install dependencies
pip install -r requirements.txt

'''

import sys, os, json, glob, shutil, random

def script_creator(output_folder, commands, time_in_mins=15000, num_gpus=8, code_dir="/fsx/ganayu/code/SuperShaper", wandb="online", wandb_entity="ganayu", wandb_project="effbert", generate_port=False, part_size=-1):
    if os.path.exists(output_folder):
        print("%s already exists"%(output_folder))
        sys.exit(0)
    print("ensure you are in login node and conda environment is activated already")
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

    # bert-pretraining data (todo: change this to your local path)
    datasets["wikibooks_graphcore_128len_next_sentence_label_removed_w_splits"] = "/fsx/ganayu/data/bert_pretraining_data/wikibooks_graphcore_128_next_sentence_label_removed_w_splits"

    return datasets[name]

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
    # where output models will be written (todo: change this to your local path)
    return "/fsx/ganayu/experiments/supershaper"

def get_code_dir():
    # where the code resides (todo: change this to your local directory)
    return "/fsx/ganayu/code/SuperShaper"

def get_model_configs(model_name):
    configs = {}
    # config file for BERT architecture (todo: change this to your local path)
    configs["bertuncased.nobottleneck.12L_768H"] = get_code_dir() + "/configs/bertbase_uncased_nobottleneck_12L_768H.csv"
    return configs[model_name]

def config_factory(name):
    configs = {}
    configs["bertbasev3.standard.train_mlm"] = {"per_device_train_batch_size": 128, "per_device_eval_batch_size": 256, "gradient_accumulation_steps": 2, "fp16": 1, "max_seq_length": 128, "mixing": "bert-bottleneck", "max_train_steps": 125000, "tokenized_c4_dir": dataset_factory("wikibooks_graphcore_128len_next_sentence_label_removed_w_splits"), "model_name_or_path": "bert-base-uncased", "sampling_type": "none", "sampling_rule": "none", "learning_rate": 5e-4, "weight_decay": 0.01, "num_warmup_steps": 10000, "eval_random_subtransformers": 0, "output_dir": "<<OUTPUT_DIR>>", "preprocessing_num_workers": 1, "betas_2": 0.98, "tokenizer_name": "Graphcore/bert-base-uncased"}
    return configs[name]

# the code requires signing up for an account in wandb: https://wandb.ai/  where all the logging metrics can be visualized (if you set wandb="online", otherwise logs will be written locally. you can later upload them into wandb.)
# todo: set wandb_entity and wandb_project after account signup
# todo: set number of GPUs and time (usually takes 20 hours on 8 GPUs)
script_creator(get_experiments_dir() + "/bert_standalone", [ {"exp_name": "standard", "runs": [{"pyfile": "train_mlm.py", "params": modify_config_and_to_string(config_factory("bertbasev3.standard.train_mlm"), {"subtransformer_config_path": get_model_configs("bertuncased.nobottleneck.12L_768H"), "initialize_pretrained_weights": "no", "mixing": "attention", "experiment_name": "bert_standalone"})}]}], time_in_mins=5000, wandb="online", wandb_entity="ganayu", wandb_project="effbert", num_gpus=8, generate_port=True, code_dir=get_code_dir())



