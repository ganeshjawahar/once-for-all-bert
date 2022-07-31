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

def get_learning_curve_fromwandb(plot_output, supernet_runids=None, standalone_runids=None, every_x_steps=-1):
    import wandb
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    api = wandb.Api()

    os.makedirs(plot_output, exist_ok=True)

    trainloss_scores, valloss_scores = {}, {}
    for name, runid in supernet_runids: 
        # attribs: Supertransformer mlm loss, Smallest mlm loss, SuperTransformer Val Loss
        run = api.run(runid)
        metrics = run.scan_history()
        if name not in trainloss_scores:
            # if name != "supernet-swichrelaxed":
            trainloss_scores[name + "-big"] = [[],[]] # step, loss
            trainloss_scores[name + "-small"] = [[],[]] # step, loss
            valloss_scores[name + "-big"] = [[],[]] # step, loss
            valloss_scores[name + "-small"] = [[],[]] # step, loss
        for row in metrics:
            if 'Supertransformer mlm loss' in row  and row['Supertransformer mlm loss'] != 'NaN': # and not np.isnan(row['Supertransformer mlm loss']):
                trainloss_scores[name + "-big"][0].append(row['_step'])
                trainloss_scores[name + "-big"][1].append(row['Supertransformer mlm loss'])
            if 'Smallest mlm loss' in row and row['Smallest mlm loss'] != 'NaN':
                trainloss_scores[name + "-small"][0].append(row['_step'])
                trainloss_scores[name + "-small"][1].append(row['Smallest mlm loss'])
            if 'SuperTransformer Val loss' in row and row['SuperTransformer Val loss'] != 'NaN': # and not np.isnan(row['SuperTransformer Val loss']):
                valloss_scores[name + "-big"][0].append(row['_step'])
                valloss_scores[name + "-big"][1].append(row['SuperTransformer Val loss'])
            if 'SmallestTransformer Val loss' in row and row['SmallestTransformer Val loss'] != 'NaN': # and not np.isnan(row['SmallestTransformer Val loss']):
                valloss_scores[name + "-small"][0].append(row['_step'])
                valloss_scores[name + "-small"][1].append(row['SmallestTransformer Val loss'])
            # for inplace kd trainings
            if 'Supertransformer Teacher mlm loss' in row  and row['Supertransformer Teacher mlm loss'] != 'NaN': # and not np.isnan(row['Supertransformer mlm loss']):
                trainloss_scores[name + "-big"][0].append(row['_step'])
                trainloss_scores[name + "-big"][1].append(row['Supertransformer Teacher mlm loss'])
            if 'Smallest Student mlm loss' in row and row['Smallest Student mlm loss'] != 'NaN':
                trainloss_scores[name + "-small"][0].append(row['_step'])
                trainloss_scores[name + "-small"][1].append(row['Smallest Student mlm loss'])
    
    for name, runid in standalone_runids: 
        # attribs: Subtransf
        # 
        # ormer mlm loss, SuperTransformer Val loss
        run = api.run(runid)
        metrics = run.scan_history()
        if name not in trainloss_scores:
             trainloss_scores[name] = [[],[]] # step, loss
             #if name != "standalone-12L-120H":
             valloss_scores[name] = [[],[]] # step, loss
        for row in metrics:
            if 'Subtransformer mlm loss' in row and row['Subtransformer mlm loss'] != 'NaN':
                trainloss_scores[name][0].append(row['_step'])
                trainloss_scores[name][1].append(row['Subtransformer mlm loss'])
            if 'SuperTransformer Val loss' in row and row['SuperTransformer Val loss'] != 'NaN': # name != "standalone-12L-120H" 
                valloss_scores[name][0].append(row['_step'])
                valloss_scores[name][1].append(row['SuperTransformer Val loss']) # todo: add subtransformer val loss

    for name, scores in [("train_loss", trainloss_scores), ("val_loss", valloss_scores)]:
        fig = plt.figure(figsize=(13,7))
        colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', "indigo", "violet", "springgreen", "olive", "firebrick", "gold"]
        ei = 0
        for model in sorted(scores):
            if name == "val_loss" or every_x_steps == -1:
                sns.lineplot(x=scores[model][0], y=scores[model][1], color=colors[ei], label=model)
            else:
                cur_x, cur_y = [], []
                for j in range(len(scores[model][0])):
                    if j % every_x_steps == 0:
                        cur_x.append(scores[model][0][j])
                        cur_y.append(scores[model][1][j])
                sns.lineplot(x=cur_x, y=cur_y, color=colors[ei], label=model)
            ei += 1
        plt.xlabel("Steps", fontsize=15)
        plt.ylabel("%s"%(name), fontsize=15)
        plt.legend(loc="upper right", ncol=2, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        fig.savefig("%s/%s.png"%(plot_output, name))

# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul18_plots", supernet_runids=[("supernet-base", "ganayu/effbert/37tipjbc"), ("supernet-swichrelaxed", "ganayu/effbert/28flai6f")], standalone_runids=[("standalone-12L-120H", "ganayu/effbert/q6fsxzge"), ("standalone-12L-768H", "ganayu/effbert/3roltlci")])
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul19_v3_plots", supernet_runids=[ ("supernet-base", "ganayu/effbert/sanqznoy") ], standalone_runids=[("standalone-12L-120H", "ganayu/effbert/2d2niusu"), ("standalone-12L-768H", "ganayu/effbert/1h79h5q7"), ("standalone-12L-768H-noinit-nobottle", "ganayu/effbert/479ja1yv")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_inplkd_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-inpkd", "ganayu/effbert/runs/2i4m1mig") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_inplkd_lasthidattn_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-inpkd-lasthidattn", "ganayu/effbert/runs/1zkvalga") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_init_backbone_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-noinit", "ganayu/effbert/runs/2jjnxy5j") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul27_extending_searchspace", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-FFNint", "ganayu/effbert/runs/2l6d5lx5"), ("supernet-hid", "ganayu/effbert/runs/83o1bxsw") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul28_2xtrainingbudget", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-1.6x", "ganayu/effbert/326b22dg")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul28_sandwch2rands", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-2rands", "ganayu/effbert/381rknsh")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul29_ffn_expan_ratio", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-ffnelastic", "ganayu/effbert/300iqwdt")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)


def wandb_locate_proj():
    import json
    # for f in glob.glob("/fsx/ganayu/code/SuperShaper/wandb/*/files/wandb-metadata.json"):
    for f in glob.glob('/fsx/ganayu/experiments/supershaper/jul19_v3_supernetbase/code/wandb*/files/wandb-metadata.json'):
        content = json.load(open(f))
        if content["gpu_count"] == 8 and content["codePath"] == "train_mlm.py":
            print(f.split("/")[-3])
# wandb_locate_proj()   


def poke_bertbase_weights():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    print(model.bert.embeddings.word_embeddings.weight[123,34:55])
    print(model.bert.embeddings.word_embeddings.weight[564,10:25])

# poke_bertbase_weights()

def get_best_ft_results():
    for model in ["google", "graph", "our"]:
        for task in ["cola", "sst2"]:
            dev_acc = 0.0
            for folder in glob.glob("/fsx/ganayu/experiments/supershaper/jul24_v3_finetune_glueoriginal_sst2_cola_toktypecorrected/*%s*%s*/sys.err"%(model, task)):
                cur_acc = None
                print(folderdw)
                for line in open(folder):
                    line = line.strip()
                    if "accuracy" in line and "epoch" in line:
                        print(line)
            sys.exit(0)
# get_best_ft_results()
