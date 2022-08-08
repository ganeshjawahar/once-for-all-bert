#!/bin/bash

# python aws/prepare_wikibooks.py
# python -c "import torch;print(torch.cuda.is_available())"
# nvidia-smi
#wandb online
#TOKENIZERS_PARALLELISM=false
#accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml train_mlm.py --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --fp16 1 --max_seq_length 128 --mixing bert-bottleneck --max_train_steps 175214 --c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 2e-5 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 0 --wandb_suffix 128seqlen --wandb_entity ganayu --wandb_project effbert --output_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen --preprocessing_num_workers 1

#export PYTHONPATH=$PYTHONPATH
#accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml supernet_ppl.py
#accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml evolution.py --params_constraints 60 --device_type gpu
#accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml evolution.py --params_constraints 60000000 --device_type gpu --population_size 2 --parent_size 2 --crossover_size 1 --mutation_size 1 --time_budget 4 --output_dir /fsx/ganayu/experiments/supershaper/jul10_search_results --per_device_eval_batch_size 400 
#cd /fsx/ganayu/code/SuperShaper/search
#export PYTHONPATH=$PYTHONPATH:../
#accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml evolution.py --params_constraints 60000000 --device_type gpu --output_dir /fsx/ganayu/experiments/supershaper/jul10_search_results --per_device_eval_batch_size 400
# accelerate launch train_glue.py --learning_rate=5e-05 --mixing=bert-bottleneck --model_name_or_path=/fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_128seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_128seqlen_08-07-2022-06-40-22/best_model --num_train_epochs=10 --per_device_train_batch_size=32 --sampling_type=none --task=mrpc --wandb_suffix finetune --wandb_entity ganayu --wandb_project effbert --subtransformer_config_path /fsx/ganayu/experiments/supershaper/jul10_search_results/gpu_params60000000.0_/best_configs_iter_299.csv --output_dir /fsx/ganayu/experiments/supershaper/jul11_finetune_results --max_train_steps 50

# cd /fsx/ganayu/code/SuperShaper
# python supernet_ppl.py

# wandb offline
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml train_mlm.py --per_device_train_batch_size 16 --per_device_eval_batch_size 128 --gradient_accumulation_steps 16 --fp16 1 --max_seq_length 512 --mixing bert-bottleneck --max_train_steps 10 --c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 0.0001 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 0 --output_dir /fsx/ganayu/experiments/supershaper/jul14_supernetbase_initial/base --betas_2 0.98 --experiment_name jul14_supernetbase_initial_base --wandb_suffix base --wandb_entity ganayu --wandb_project effbert --preprocessing_num_workers 32
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_mn_config.yaml train_mlm.py --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 --fp16 1 --max_seq_length 512 --mixing bert-bottleneck --tokenized_c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final_tokenized --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 0.0001 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 0 --output_dir /fsx/ganayu/experiments/supershaper/jul14_supernetbase_initial/base --preprocessing_num_workers 1 --betas_2 0.98 --experiment_name jul14_supernetbase_initial_base --wandb_suffix base --wandb_entity ganayu --wandb_project effbert
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_mn_config.yaml train_mlm.py --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --gradient_accumulation_steps 4 --fp16 1 --max_seq_length 512 --mixing bert-bottleneck --tokenized_c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy_tokenized --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 0.0001 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 0 --output_dir /fsx/ganayu/experiments/supershaper/jul14_supernetbase_initial/base --preprocessing_num_workers 1 --betas_2 0.98 --experiment_name jul14_supernetbase_initial_base --wandb_suffix base --wandb_entity ganayu --wandb_project effbert
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_mn_config.yaml --main_process_port 22128 evo_ours.py --per_device_eval_batch_size 512 --mixing bert-bottleneck --supernet_ckpt_dir /fsx/ganayu/experiments/supershaper/jul19_v3_supernetbase/base/best_model --data_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_graphcore_128_next_sentence_label_removed_w_splits --output_dir /fsx/ganayu/experiments/trial

# cd /fsx/ganayu/code/SuperShaper/academic-bert-dataset
# python generate_samples.py --dir /fsx/ganayu/data/academic_bert_dataset/shardoutput -o /fsx/ganayu/data/academic_bert_dataset/roberta_final_samples --dup_factor 1 --seed 42 --vocab_file roberta-base --do_lower_case 1 --masked_lm_prob 0.15 --max_seq_length 128 --model_name roberta-base --max_predictions_per_seq 20 --n_processes 1
python aws/prepare_wikibooks.py /fsx/ganayu/data/academic_bert_dataset/roberta_final_samples /fsx/ganayu/data/academic_bert_dataset/roberta_preproc_128
