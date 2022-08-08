#!/bin/bash

# fetch c4-realnews
# python aws/data_cleaning.py
# python aws/trial.py
# python aws/prepare_wikibooks.py
# cd /fsx/ganayu/data/academic_bert_dataset
# wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# python academic-bert-dataset/process_data.py -f /fsx/ganayu/data/academic_bert_dataset/enwiki-latest-pages-articles.xml.bz2.1.out -o /fsx/ganayu/data/academic_bert_dataset/wikiproc --type wiki
# wget https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz
# python academic-bert-dataset/process_data.py -f /fsx/ganayu/data/academic_bert_dataset/books_raw/books1/epubtxt -o /fsx/ganayu/data/academic_bert_dataset/booksproc --type bookcorpus
# python academic-bert-dataset/shard_data.py --dir /fsx/ganayu/data/academic_bert_dataset/wikibooksproc -o /fsx/ganayu/data/academic_bert_dataset/shardoutput --num_train_shards 256 --num_test_shards 128 --frac_test 0.1
# cd /fsx/ganayu/code/SuperShaper/academic-bert-dataset
# python generate_samples.py --dir /fsx/ganayu/data/academic_bert_dataset/shardoutput -o /fsx/ganayu/data/academic_bert_dataset/final_samples --dup_factor 1 --seed 42 --vocab_file bert-base-uncased --do_lower_case 1 --masked_lm_prob 0.15 --max_seq_length 128 --model_name bert-base-uncased --max_predictions_per_seq 20 --n_processes 4
# python create_pretraining_data.py --input_file=/fsx/ganayu/data/academic_bert_dataset/shardoutput --output_file=/fsx/ganayu/data/academic_bert_dataset/final_samples --vocab_file=bert-base-uncased --bert_model=bert-base-uncased --do_lower_case --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=42 --dupe_factor=1 --no_nsp
python aws/prepare_wikibooks.py /fsx/ganayu/data/academic_bert_dataset/final_samples /fsx/ganayu/data/academic_bert_dataset/bert_preproc_128

# supernet pretrain
# python aws/trial.py
# accelerate launch train_mlm.py --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --fp16 1 --max_seq_length 128 --mixing bert-bottleneck --max_train_steps 175214 --c4_dir /fsx/ganayu/data/supershaper_pretraining_data/c4_datasets --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 5e-5 --weight_decay 0.0 --num_warmup_steps 0 --eval_random_subtransformers 1 --wandb_suffix trial --wandb_entity ganayu --wandb_project effbert --output_dir /fsx/ganayu/experiments/supershaper/june28_trial
# wandb offline
# TOKENIZERS_PARALLELISM=false
# accelerate launch train_mlm.py --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --fp16 1 --max_seq_length 128 --mixing bert-bottleneck --max_train_steps 175214 --c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 2e-5 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 1 --wandb_suffix trial --wandb_entity ganayu --wandb_project effbert --output_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_v2 --preprocessing_num_workers 1 --inplace_distillation 1
# wandb online
#accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml train_mlm.py --per_device_train_batch_size 128 --per_device_eval_batch_size 256 --gradient_accumulation_steps 2 --fp16 1 --max_seq_length 512 --mixing bert-bottleneck --max_train_steps 175214 --c4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_final --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 2e-5 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 0 --wandb_suffix 512seqlen --wandb_entity ganayu --wandb_project effbert --output_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen --preprocessing_num_workers 1
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml evolution.py --params_constraints 60000000 --device_type gpu --output_dir /fsx/ganayu/experiments/supershaper/jul10_search_results --per_device_eval_batch_size 400  --max_seq_length 512 --supernet_ckpt_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_512seqlen_08-07-2022-06-48-14/best_model
# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml evolution.py --params_constraints 60000000 --device_type gpu --output_dir /fsx/ganayu/experiments/supershaper/jul10_search_results_512seqlen --per_device_eval_batch_size 400  --max_seq_length 512 --supernet_ckpt_dir /fsx/ganayu/experiments/supershaper/jul6_bertdata_bertbottleneck_512seqlen/c4_realnews_bert-bottleneck_random_K=1_pretraining_512seqlen_08-07-2022-06-48-14/best_model

# accelerate launch --config_file /fsx/ganayu/code/SuperShaper/default_config.yaml train_mlm.py --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --gradient_accumulation_steps 8 --fp16 1 --max_seq_length 512 --mixing bert-bottleneck --Ωc4_dir /fsx/ganayu/data/bert_pretraining_data/wikibooks_datasets_dummy --model_name_or_path bert-base-cased --sampling_type random --sampling_rule sandwich --learning_rate 0.0001 --weight_decay 0.01 --num_warmup_steps 10000 --eval_random_subtransformers 0 --output_dir /fsx/ganayu/experiments/supershaper/jul14_supernetbase_initial/base --preprocessing_num_workers 16 --betas_2 0.98 --experiment_name jul14_supernetbase_initial_base --wandb_suffix base --wandb_entity ganayu --wandb_project effbert




