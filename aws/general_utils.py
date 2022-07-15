'''
all general utilities
'''

import sys, os, json, glob


def check_autotinybert_model_size():
    # from transformers import BERT, AutoModelForMaskedLM
    # tokenizer = AutoTokenizer.from_pretrained("huawei-noah/AutoTinyBERT-KD-S1", use_fast=True)
    # model = AutoModelForMaskedLM.from_pretrained("huawei-noah/AutoTinyBERT-S2", cache_dir="/fsx/ganayu/experiments/supershaper/cache")
    # from transformers import BertConfig, BertTokenizer, BertForPreTraining, BertForMaskedLM
    # config = BertConfig.from_pretrained('bert-base-uncased', cache_dir="/fsx/ganayu/experiments/supershaper/cache")
    # print(config)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/fsx/ganayu/experiments/supershaper/cache")
    # model = BertForPreTraining.from_pretrained('bert-base-uncased', cache_dir="/fsx/ganayu/experiments/supershaper/cache")
    # model = BertForMaskedLM.from_pretrained(config, cache_dir="/fsx/ganayu/experiments/supershaper/cache")
    # print(sum(p.numel() for p in model.parameters()))

    def calc_params(emb_size=564, hidden_size=564, qkv_size=512, intermediate_size=1054, max_position_embeddings=512, num_attention_heads=8, num_hidden_layers=5, type_vocab_size=2, vocab_size=30522):
        emb_params = (vocab_size + 512 + 2) * emb_size + 2 * emb_size
        # per_layer_params = 0
        # for layer_idx, (d_ff, emb_dim) in enumerate(zip([intermediate_size]*num_hidden_layers, emb_dims)):
        # embeddings_params = VOCAB + POSITION + TYPE + LAYERNORM
        embeddings_params = (vocab_size * hidden_size) + (max_position_embeddings * hidden_size) + (2 * hidden_size) + (2 * hidden_size)
        # encoder_params = HIDDEN_LAYERS * ( (ATTENTION HEAD) + (FFN) + (LayerNorm))
        encoder_params = num_hidden_layers * ((4 * (hidden_size * qkv_size) + qkv_size) + (2 * (hidden_size * intermediate_size) + hidden_size + intermediate_size) + (2 * hidden_size))
        pooler_params = hidden_size * hidden_size
        print(embeddings_params + encoder_params + pooler_params)

    calc_params() # https://huggingface.co/huawei-noah/AutoTinyBERT-S1/tree/main # 29M
    # calc_params(emb_size=564, hidden_size=564, qkv_size=528, intermediate_size=1024, max_position_embeddings=512, num_attention_heads=12, num_hidden_layers=5, type_vocab_size=2, vocab_size=30522) # AutoTinyBERT-KD-S1 # 29M
    calc_params(emb_size=396, hidden_size=396, qkv_size=384, intermediate_size=624, max_position_embeddings=512, num_attention_heads=6, num_hidden_layers=4, type_vocab_size=2, vocab_size=30522) # AutoTinyBERT-S2 # 16M

# check_autotinybert_model_size()

def get_finetuning_results(folder):
    res = ""
    scores = []
    for task in ["mnli", "qqp",  "qnli", "cola", "sst2", "stsb", "rte", "mrpc"]:
        sysout_f = glob.glob(folder + "/*%s/sys.out"%task)
        if len(sysout_f) == 1:
            cur_score = -1.0
            for line in open(sysout_f[0]):
                line = line.strip()
                if task + " epoch " in line and "best" in line:
                    cur_score = float(line.split()[-1])
            scores.append(cur_score)
            res += "%.1f,"%(100.0*cur_score)
        else:
            scores.append(0.0)
            res += "-,"
    print(scores)
    import numpy as np
    avg_score = np.mean(scores)
    nasbertscore = 83.5
    res += "%.1f (%.1f%%)"%(100.0*avg_score, 100.0*(((100.0*avg_score)-nasbertscore)/nasbertscore))
    print(res)

# get_finetuning_results("/fsx/ganayu/experiments/supershaper/jul11_wikibooks_finetune_9tasks")
# get_finetuning_results("/fsx/ganayu/experiments/supershaper/jul12_wikibooks_efficient_subnet_train_more_steps_finetune_9tasks")
# get_finetuning_results("/fsx/ganayu/experiments/supershaper/jul12_wikibooks_efficient_supernet_train_more_steps_finetune_9tasks")
# get_finetuning_results("/fsx/ganayu/experiments/supershaper/jul12_wikibooks_lr5e-5_finetune_9tasks")
# get_finetuning_results("/fsx/ganayu/experiments/supershaper/jul12_wikibooks_efficient_subnet_train_more_steps_100K_finetune_9tasks")
