#!/bin/bash

# fetch c4-realnews
# python aws/data_cleaning.py
# python aws/trial.py

# supernet pretrain
# python aws/trial.py
# accelerate launch train_mlm.py --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --fp16 1 --max_seq_length 128 --mixing bert-bottleneck --max_train_steps 175214 --c4_dir /fsx/ganayu/data/supershaper_pretraining_data/c4_datasets --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 5e-5 --weight_decay 0.0 --num_warmup_steps 0 --eval_random_subtransformers 1 --wandb_suffix trial --wandb_entity ganayu --wandb_project effbert --output_dir /fsx/ganayu/experiments/supershaper/june28_trial
# wandb offline
TOKENIZERS_PARALLELISM=false
# accelerate launch train_mlm.py --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --fp16 1 --max_seq_length 128 --mixing bert-bottleneck --max_train_steps 175214 --c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 2e-5 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 1 --wandb_suffix trial --wandb_entity ganayu --wandb_project effbert --output_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_v2 --preprocessing_num_workers 1 --inplace_distillation 1
# wandb online
#accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml train_mlm.py --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --fp16 1 --max_seq_length 512 --mixing bert-bottleneck --max_train_steps 175214 --c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 2e-5 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 0 --wandb_suffix 512seqlen --wandb_entity ganayu --wandb_project effbert --output_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen --preprocessing_num_workers 1
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml evolution.py --params_constraints 60000000 --device_type gpu --output_dir /fsx/ganayu/experiments/supershaper/jul10_search_results --per_device_eval_batch_size 400  --max_seq_length 512 --supernet_ckpt_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_512seqlen_08-07-2022-06-48-14/best_model
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml evolution.py --params_constraints 60000000 --device_type gpu --output_dir /fsx/ganayu/experiments/supershaper/jul10_search_results_512seqlen --per_device_eval_batch_size 400  --max_seq_length 512 --supernet_ckpt_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_512seqlen_08-07-2022-06-48-14/best_model