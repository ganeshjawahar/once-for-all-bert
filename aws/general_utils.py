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

def get_learning_curve_fromwandb(plot_output, supernet_runids=None, standalone_runids=None, every_x_steps=-1, inpl_kd=None, ignore_first_x_steps=None):
    import wandb
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.set_theme(style="darkgrid")
    api = wandb.Api()

    os.makedirs(plot_output, exist_ok=True)

    trainloss_scores, valloss_scores = {}, {}
    if inpl_kd:
        if "logits" in inpl_kd:
            distillloss_scores = {}
        if "hidden" in inpl_kd:
            hiddenloss_scores = {}
        if "attention" in inpl_kd:
            attentionloss_scores = {}
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
            if inpl_kd:
                if "logits" in inpl_kd:
                    distillloss_scores[name + "-small"] = [[],[]] # step, loss
                    distillloss_scores[name + "-rand"] = [[],[]] # step, loss
                if "hidden" in inpl_kd:
                    hiddenloss_scores[name + "-small"] = [[],[]] # step, loss
                    hiddenloss_scores[name + "-rand"] = [[],[]] # step, loss
                if "attention" in inpl_kd:
                    attentionloss_scores[name + "-small"] = [[],[]] # step, loss
                    attentionloss_scores[name + "-rand"] = [[],[]] # step, loss
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
            if inpl_kd:
                if 'Smallest Student distill loss' in row and row['Smallest Student distill loss'] != 'NaN':
                    distillloss_scores[name + "-small"][0].append(row['_step'] if '_step' in row else len(distillloss_scores[name + "-small"][0]))
                    distillloss_scores[name + "-small"][1].append(row['Smallest Student distill loss'])
                if 'Subtransformer Student distill loss' in row and row['Subtransformer Student distill loss'] != 'NaN':
                    distillloss_scores[name + "-rand"][0].append(row['_step'] if '_step' in row else len(distillloss_scores[name + "-rand"][0]))
                    distillloss_scores[name + "-rand"][1].append(row['Subtransformer Student distill loss'])
                if 'Smallest Student hidden loss' in row and row['Smallest Student hidden loss'] != 'NaN':
                    hiddenloss_scores[name + "-small"][0].append(row['_step'] if '_step' in row else len(hiddenloss_scores[name + "-small"][0]))
                    hiddenloss_scores[name + "-small"][1].append(row['Smallest Student hidden loss'])
                if 'Subtransformer Student hidden loss' in row and row['Subtransformer Student hidden loss'] != 'NaN':
                    hiddenloss_scores[name + "-rand"][0].append(row['_step'] if '_step' in row else len(hiddenloss_scores[name + "-rand"][0]))
                    hiddenloss_scores[name + "-rand"][1].append(row['Subtransformer Student hidden loss'])
                if 'Smallest Student attention loss' in row and row['Smallest Student attention loss'] != 'NaN':
                    attentionloss_scores[name + "-small"][0].append(row['_step'] if '_step' in row else len(attentionloss_scores[name + "-small"][0]))
                    attentionloss_scores[name + "-small"][1].append(row['Smallest Student attention loss'])
                if 'Subtransformer Student attention loss' in row and row['Subtransformer Student attention loss'] != 'NaN':
                    attentionloss_scores[name + "-rand"][0].append(row['_step'] if '_step' in row else len(attentionloss_scores[name + "-rand"][0]))
                    attentionloss_scores[name + "-rand"][1].append(row['Subtransformer Student attention loss'])
    
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
    
    scores_list = [("train_loss", trainloss_scores), ("val_loss", valloss_scores)]
    if inpl_kd:
        if "logits" in inpl_kd:
            scores_list.append(("distill_loss", distillloss_scores))
        if "hidden" in inpl_kd:
            scores_list.append(("hidden_loss", hiddenloss_scores))
        if "attention" in inpl_kd:
            scores_list.append(("attention_loss", attentionloss_scores))
    good_names = {"arch experts-big": "Arch. Experts (Big)", "arch experts-small": "Arch. Experts (Small)", "neuron experts-big": "Neuron Experts (Big)", "neuron experts-small": "Neuron Experts (Small)", "standalone-big": "Standalone (Big)", "standalone-small": "Standalone (Small)", "supernet-big": "Supernet (Big)", "supernet-small": "Supernet (Small)"}
    good_names = {"arch experts-big": "Layer-wise MoS (Big)", "arch experts-small": "Layer-wise MoS (Small)", "neuron experts-big": "Neuron-wise MoS (Big)", "neuron experts-small": "Neuron-wise MoS (Small)", "standalone-big": "Standalone (Big)", "standalone-small": "Standalone (Small)", "supernet-big": "Supernet (Big)", "supernet-small": "Supernet (Small)"}
    for name, scores in scores_list:
        fig = plt.figure(figsize=(13,7))
        colors = ['b', "springgreen", "indigo", "olive", "firebrick", 'c', "gold", "violet", 'm', 'r', 'g', 'k', 'y', 'fuchsia', 'maroon', 'sienna', 'orange', 'coral']
        ei = 0
        for model in sorted(scores):
            if name == "val_loss" or every_x_steps == -1:
                sns.lineplot(x=scores[model][0], y=scores[model][1], color=colors[ei], label=good_names[model])
            else:
                cur_x, cur_y = [], []
                print(name, model)
                for j in range(len(scores[model][0])):
                    if j % every_x_steps == 0: # and (ignore_first_x_steps is None or int(model[0][j]) >= ignore_first_x_steps):
                        cur_x.append(scores[model][0][j])
                        cur_y.append(scores[model][1][j])
                sns.lineplot(x=cur_x, y=cur_y, color=colors[ei], label=good_names[model])
            ei += 1
        plt.xlabel("Steps", fontsize=18)
        plt.ylabel("%s"%(name) if "val" not in name else "Validation MLM Loss", fontsize=18)
        plt.legend(loc="upper right", ncol=2, fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        fig.savefig("%s/%s.png"%(plot_output, name))

# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul18_plots", supernet_runids=[("supernet-base", "ganayu/effbert/37tipjbc"), ("supernet-swichrelaxed", "ganayu/effbert/28flai6f")], standalone_runids=[("standalone-12L-120H", "ganayu/effbert/q6fsxzge"), ("standalone-12L-768H", "ganayu/effbert/3roltlci")])
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul19_v3_plots", supernet_runids=[ ("supernet-base", "ganayu/effbert/sanqznoy") ], standalone_runids=[("standalone-12L-120H", "ganayu/effbert/2d2niusu"), ("standalone-12L-768H", "ganayu/effbert/1h79h5q7"), ("standalone-12L-768H-noinit-nobottle", "ganayu/effbert/479ja1yv")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_inplkd_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-inpkd", "ganayu/effbert/runs/2i4m1mig") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_inplkd_lasthidattn_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-inpkd-lasthidattn", "ganayu/effbert/runs/1zkvalga") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul24_fastai_pres_init_backbone_plots", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-noinit", "ganayu/effbert/runs/2jjnxy5j") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul27_extending_searchspace", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-FFNint", "ganayu/effbert/runs/2l6d5lx5"), ("supernet-hid", "ganayu/effbert/runs/83o1bxsw") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul28_2xtrainingbudget", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-2x", "ganayu/effbert/326b22dg")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul28_sandwch2rands", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-2rands", "ganayu/effbert/381rknsh")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jul29_ffn_expan_ratio", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-ffnelastic", "ganayu/effbert/300iqwdt")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug1_inplkd_lossscaling", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-inkd-scale1.0", "ganayu/effbert/21i6ad9y"), ("supernet-inkd-scale1.5", "ganayu/effbert/3a6nwrht"), ("supernet-inkd-scale2.0", "ganayu/effbert/3e65ue8y")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100, inpl_kd=["logits"], ignore_first_x_steps=50000)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug1_inplkd_logits_hidden", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"),  ("supernet-inkd-logits=soft", "ganayu/effbert/21i6ad9y"), ("supernet-inkd-logits=hard+soft", "ganayu/effbert/1m7t4p8f"), ("supernet-inkd-logits=soft+hidden", "ganayu/effbert/1yh9dkl0")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100, inpl_kd=["logits", "hidden"], ignore_first_x_steps=50000)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug1_hypernet", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-hypnet-rank32-hyphid16", "ganayu/effbert/283vgn33"), ("supernet-hypnet-rank64-hyphid16", "ganayu/effbert/1m37fnbo"), ("supernet-hypnet-rank64-hyphid50", "ganayu/effbert/jrmtzw1h") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100, ignore_first_x_steps=50000)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug1_nasbert_bsz_1024_250Ksteps", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("supernet-1024_250Ksteps", "ganayu/effbert/24lo78gh") ], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug2_continue_pretrain_100ksteps_schedule", supernet_runids=[], standalone_runids=[("stage1", "ganayu/effbert/oisadmiy"), ("5timeslowerlr", "ganayu/effbert/39p9dmtg"), ("nowarmup", "ganayu/effbert/2m79af3n")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug8_continue_pretrain_100ksteps_schedule_softhard_vs_hard", supernet_runids=[], standalone_runids=[("soft+hard", "ganayu/effbert/3emvtqgq"), ("hard", "ganayu/effbert/2m79af3n")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug8_inplkd_logits_hidden", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"),  ("attentionhard", "ganayu/effbert/2gj82hqh"), ("hiddenhard", "ganayu/effbert/1ho5yd71"), ("logitshard", "ganayu/effbert/v8by4dk4")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], inpl_kd=["logits", "hidden", "attention"], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug10_v1_vs_v3", supernet_runids=[ ("supernet-v1", "ganayu/effbert/sanqznoy"), ("supernet-v3", "ganayu/effbert/2amuc50v") ], standalone_runids=[], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug20_inplkd_logits_attention", supernet_runids=[ ("supernet", "ganayu/effbert/sanqznoy"), ("mlm1-att1", "ganayu/effbert/3igyemgd"), ("mlm0.5-logit0.5", "ganayu/effbert/14jy3peu")], standalone_runids=[("standalone-small", "ganayu/effbert/2d2niusu"), ("standalone-big", "ganayu/effbert/1h79h5q7")], inpl_kd=["attention", "logits"], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug23_subnet_kd", supernet_runids=[], standalone_runids=[("MLM+hiddenproj", "ganayu/effbert/tcr8v07b"), ("MLM+hid", "ganayu/effbert/1h79h5q7"), ("MLM+hid", "ganayu/effbert/i705bfim"), ("MLM+att", "ganayu/effbert/3npbclo0")], inpl_kd=["attention"], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug24_subnet_kd", supernet_runids=[], standalone_runids=[("scratch-MLM", "ganayu/effbert/1kjx2mpp"), ("continue-MLM+hid24681012", "ganayu/effbert/vnw397qh"), ("continue-MLM+hid12", "ganayu/effbert/2t12zawf"), ("continue-MLM+hid1-12", "ganayu/effbert/7r2twr3j"), ("continue-MLM+logits", "ganayu/effbert/rztz6cia"), ("continue-MLM", "ganayu/effbert/igurztly") ], inpl_kd=["attention"], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug24_subnet_kd", supernet_runids=[], standalone_runids=[("scratch-MLM", "ganayu/effbert/1kjx2mpp"),  ("continue-MLM+hid24681012", "ganayu/effbert/vnw397qh"), ("continue-MLM+hid12", "ganayu/effbert/2t12zawf"), ("continue-MLM+hid1-12", "ganayu/effbert/7r2twr3j"), ("continue-MLM+logits", "ganayu/effbert/rztz6cia"), ("continue-MLM", "ganayu/effbert/igurztly"), ("scratch+logits-MLM", "ganayu/effbert/3by8zjll") ], inpl_kd=["attention"], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug26_supernet_inplkd", supernet_runids=[ ("supernet", "ganayu/effbert/2hismi0h"), ("MLM+hid1-12", "ganayu/effbert/1rlq5v50"), ("MLM+hid12", "ganayu/effbert/3inltdpj"), ("MLM+hid24681012", "ganayu/effbert/ocql4hmv"), ("MLM+logits", "ganayu/effbert/2c18e6jq"), ("MLM+goologits", "ganayu/effbert/2jfrxvav")], standalone_runids=[("standalone-big", "ganayu/effbert/2yyuo4mm"), ("standalone-small", "ganayu/effbert/39bn06ci")], inpl_kd=["hidden", "logits"], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug31_supernet_moe", supernet_runids=[ ("supernet", "ganayu/effbert/2hismi0h"), ("4e_1rand", "ganayu/effbert/dns0jab5"), ("2e_1rand", "ganayu/effbert/1we400qi")], standalone_runids=[("standalone-big", "ganayu/effbert/2yyuo4mm"), ("standalone-small", "ganayu/effbert/39bn06ci")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug31_supernet_moe_2rand", supernet_runids=[ ("4e_1rand", "ganayu/effbert/dns0jab5"), ("2e_1rand", "ganayu/effbert/1we400qi"), ("2e_2rand", "ganayu/effbert/x80tlkv1"), ("4e_2rand", "ganayu/effbert/319k7x8y")], standalone_runids=[], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/sep7_supernet_moe_jack", supernet_runids=[ ("2e", "ganayu/effbert/1we400qi"), ("2e+jack", "ganayu/effbert/i3w3y04e")], standalone_runids=[], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/oct18_supernet_archexperts", supernet_runids=[ ("supernet", "ganayu/effbert/2hismi0h"), ("randmoe", "ganayu/effbert/1we400qi"), ("archmoe_pavg_1L", "ganayu/effbert/runs/numd2vy3"),  ("archmoe_sinexp", "ganayu/effbert/runs/2avuv91k")], standalone_runids=[], every_x_steps=100) # , standalone_runids=[("standalone-big", "ganayu/effbert/2yyuo4mm"), ("standalone-small", "ganayu/effbert/39bn06ci")] ("archmoe_pavg_2L", "ganayu/effbert/runs/nwito9f8"),
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/oct21_supernet_archexperts", supernet_runids=[ ("supernet", "ganayu/effbert/2hismi0h"), ("archmoe_hid64", "ganayu/effbert/5ou0wyus"), ("fixedarchmoe_hid64", "ganayu/effbert/6d5mmpk7")], standalone_runids=[("standalone-big", "ganayu/effbert/2yyuo4mm"), ("standalone-small", "ganayu/effbert/39bn06ci")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/nov15_supernet_archexperts", supernet_runids=[ ("supernet", "ganayu/effbert/2hismi0h"), ("arch experts", "ganayu/effbert/92ojue4d"), ("neuron experts", "ganayu/effbert/20fwsb8z") ], standalone_runids=[("standalone-big", "ganayu/effbert/2yyuo4mm"), ("standalone-small", "ganayu/effbert/39bn06ci")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/dec6_supernet_archexperts_difflr", supernet_runids=[ ("supernet", "ganayu/effbert/2hismi0h"), ("neuronexp5e-4", "ganayu/effbert/20fwsb8z"), ("neuronexp8e-4", "ganayu/effbert/1vr5wzx5") ], standalone_runids=[("standalone-big", "ganayu/effbert/2yyuo4mm"), ("standalone-small", "ganayu/effbert/39bn06ci")], every_x_steps=100)
# get_learning_curve_fromwandb(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/jan10_paper_supernet_archexperts", supernet_runids=[ ("supernet", "ganayu/effbert/2hismi0h"), ("arch experts", "ganayu/effbert/92ojue4d"), ("neuron experts", "ganayu/effbert/20fwsb8z") ], standalone_runids=[("standalone-big", "ganayu/effbert/2yyuo4mm"), ("standalone-small", "ganayu/effbert/39bn06ci")], every_x_steps=100)


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
                print(folder)
                for line in open(folder):
                    line = line.strip()
                    if "accuracy" in line and "epoch" in line:
                        print(line)
            sys.exit(0)
# get_best_ft_results()

def get_best_mnli_ft_results(experiments, models, tasks, mm=False):
    for model in models: #[ "2xtrainbudget"]: #"v2", "2xbudget", "sandwch_2rand",]:
        for task in tasks: #["cola", "mrpc", "sst2"]:
            for exp in experiments:
                best_mnli_dev_acc = None
                best_config = None
                for f in glob.glob("/fsx/ganayu/experiments/supershaper/%s/*%s*%s*/sys.out"%(exp, model, task)):
                    last_acc = None
                    for line in open(f):
                        line = line.strip()
                        if mm and task == "mnli":
                            if "accuracy" in line:
                                last_acc = 100.0*float(line.split("'accuracy': ")[1].split("}")[0])
                        else:
                            if "accuracy" in line and "lr=" in line:
                                if "'f1'" in line:
                                    last_acc = 100.0*float(line.split("'accuracy': ")[1].split(",")[0])
                                else:
                                    last_acc = 100.0*float(line.split("'accuracy': ")[1].split("}")[0])
                            elif "spearmanr" in line and "lr=" in line:
                                last_acc = 100.0*float(line.split("'spearmanr': ")[1].split("}")[0])
                            elif "matthews_correlation" in line:
                                last_acc = 100.0*float(line.split("'matthews_correlation': ")[1].split("}")[0])
                            elif "Accuracy" in line:
                                last_acc = 100.0*float(line.split("Accuracy': ")[1].split("}")[0])
                    if last_acc:
                        if best_mnli_dev_acc is None or best_mnli_dev_acc < last_acc:
                            best_mnli_dev_acc = last_acc
                            best_config = f
                print(model, task, best_mnli_dev_acc, best_config)

# get_best_mnli_ft_results(["aug2_v3_finetune_supershaper_60M_direct", "aug2_v3_finetune_supershaper_60M_100Ksteps", "aug2_v3_finetune_v2_60M_direct"])
# get_best_mnli_ft_results(["aug5_v3_finetune_sandwch_2rand_60M_100Ksteps"])
# get_best_mnli_ft_results(["aug5_v3_finetune_v1_60M_100Ksteps_cola_mrpc_sst2"])
# get_best_mnli_ft_results(["aug5_v3_finetune_2xbudget_60M_100Ksteps_cola_mrpc_sst2"])
# get_best_mnli_ft_results(["aug7_finetune_v1_60M_100Ksteps_logitshard_mnli"], models=["v1"], tasks=["mnli"])
# get_best_mnli_ft_results(["aug6_finetune_v2_60M_directFT_cola_mrpc_sst2"], models=["v2"], tasks=["cola", "mrpc", "sst2"])
# get_best_mnli_ft_results(["aug8_finetune_inplacekd_logitshard_mnli"], models=["logitshard"], tasks=["mnli"])
# get_best_mnli_ft_results(["aug8_finetune_search_v1_duplissueresolved_mnli"], models=["logitshard"], tasks=["mnli"])
# get_best_mnli_ft_results(["aug10_finetune_acabertdata_bertbasestandalone"], models=["bertbase"], tasks=["mnli", "cola", "mrpc", "sst2"], mm=True)
# get_best_mnli_ft_results(["aug10_finetune_acabertdata_robertbasestandalone"], models=["robertabase"], tasks=["mnli", "cola", "mrpc", "sst2"], mm=False)
# get_best_mnli_ft_results(["aug12_directfinetune_v1_acabert_evo15"], models=["v1"], tasks=["mnli", "cola"], mm=False)
# get_best_mnli_ft_results(["aug13_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_retrain_scratch"], models=["v1"], tasks=["mnli", "cola"], mm=False)
# get_best_mnli_ft_results(["aug13_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_supernet_continue"], models=["v1"], tasks=["mnli", "cola"], mm=False)
# get_best_mnli_ft_results(["aug13_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_supernet_continue_mnli_bestckpt_rte_stsb_mrpc"], models=["v1"], tasks=["rte", "mrpc", "stsb"], mm=False)
# get_best_mnli_ft_results(["aug19_finetune_v1sub_cola_hardhidden"], models=["v1"], tasks=["cola"], mm=False)
# get_best_mnli_ft_results(["aug18_finetune_acabertdata_bertbasestandalone_mnli_ckptneeded"], models=["bertbase"], tasks=["cola", "mnli"], mm=False)
# get_best_mnli_ft_results(["aug23_finetune_v1sub_cola_hardlogits"], models=["hardlogits"], tasks=["cola"], mm=False)
# get_best_mnli_ft_results(["aug23_finetune_v1sub_cola_hardattn"], models=["hardattn"], tasks=["cola"], mm=False)
# get_best_mnli_ft_results(["aug23_finetune_v1sub_mnli_hardlogits"], models=["hardlogits"], tasks=["mnli"], mm=False)
# get_best_mnli_ft_results(["aug22_finetune_v1sub_mnli_hardattn"], models=["hardattn"], tasks=["mnli"], mm=False)
# get_best_mnli_ft_results(["aug18_finetune_acabertdata_bertbasestandalone_sst2_qnli_qqp_ckptneeded"], models=["bertbase"], tasks=["sst2", "qnli", "qqp"], mm=False)
# get_best_mnli_ft_results(["aug18_finetune_acabertdata_bertbasestandalone_mrpc_rte_stsb_ckptneeded"], models=["bertbase"], tasks=["mrpc", "rte", "stsb"], mm=False)
# for task in ["sst2", "qnli", "qqp"]:
#  get_best_mnli_ft_results(["aug23_finetune_v1sub_%s_hardlogits"%task], models=[task], tasks=[task], mm=False)
# for task in ["mrpc", "rte"]:
#    get_best_mnli_ft_results(["aug23_finetune_v1sub_%s_hardlogits"%task], models=[task], tasks=[task], mm=False)
# get_best_mnli_ft_results(["aug24_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_supernet_continue_v3"], models=["acav3"], tasks=["mrpc", "sst2", "rte"], mm=False)
# get_best_mnli_ft_results(["aug25_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_supernet_continue"], models=[""], tasks=["mnli", "cola", "sst2"], mm=False)
# get_best_mnli_ft_results(["aug25_finetune_acabertdata_bertbasestandalone_mrpc_rte_ckptneeded"], models=[""], tasks=["mrpc", "rte"], mm=False)
# get_best_mnli_ft_results(["aug25_finetune_aca_v1subnet_rsub_60M"], models=[""], tasks=["cola", "mnli"], mm=False)
# get_best_mnli_ft_results(["aug25_directfinetune_v1sub_cola_hardlogits"], models=[""], tasks=["cola"], mm=False)
# get_best_mnli_ft_results(["aug25_directfinetune_v1sub_mnli_hardlogits"], models=[""], tasks=["mnli"], mm=False)
# get_best_mnli_ft_results(["aug25_directfinetune_v1scratch_cola_hardlogits"], models=[""], tasks=["cola"], mm=False)
# get_best_mnli_ft_results(["aug25_directfinetune_v1scratch_mnli_hardlogits"], models=[""], tasks=["mnli"], mm=False)
# get_best_mnli_ft_results(["aug25_finetune_googlebert_mnli_cola_ckptneeded"], models=[""], tasks=["mnli", "cola", "sst2", "qnli", "qqp"], mm=False)
# get_best_mnli_ft_results(["aug25_finetune_googlebert_mrpc_rte_stsb_ckptneeded"], models=[""], tasks=["mrpc", "rte"], mm=False)
# get_best_mnli_ft_results(["sep17_directfinetune_v1sub_mnli_hardlogits"], models=[""], tasks=["mnli",], mm=False)
# get_best_mnli_ft_results(["sep18_finetune_v1sub_moe_mnli_hardlogits"], models=[""], tasks=["mnli",], mm=False)

def get_best_mnli_superft_results(experiments, models, tasks, mm=False):
    for model in models: 
        for task in tasks: #["cola", "mrpc", "sst2"]:
            for exp in experiments:
                best_mnli_dev_acc, best_setting = None, None
                for f in glob.glob("/fsx/ganayu/experiments/supershaper/%s/*%s*%s*/sys.out"%(exp, model, task)):
                    last_acc = None
                    for line in open(f):
                        line = line.strip()
                        if "SmallestTransformer Val Accuracy" in line:
                            small_acc = 100.0*float(line.split("'SmallestTransformer Val Accuracy': ")[1].split("}")[0])
                            big_acc = 100.0*float(line.split("'SuperTransformer Val Accuracy': ")[1].split(",")[0])
                            last_acc = (small_acc + big_acc)/2.0
                    if best_mnli_dev_acc is None or best_mnli_dev_acc < last_acc:
                        best_mnli_dev_acc = last_acc
                        best_setting = f.split("/")[-2]
                print(model, task, best_mnli_dev_acc, best_setting)
# get_best_mnli_superft_results(["aug11_supernet_finetune_mnli_v3_bertbaseinit"], models=["bertbase"], tasks=["mnli"], mm=False) # 1_bertbase_mnli_5e-5_16_3
# get_best_mnli_superft_results(["aug16_supernet_finetune_mnli_v3_supernetbertbaseinit"], models=["bertbase"], tasks=["mnli"], mm=False) # 2_bertbase_mnli_5e-5_16_4

def get_best_ft_results_for_more_tasks(master_folder):
    '''
    for exp in glob.glob(master_folder + "/*/sys.out"):
        imp_lines = []
        for line in open(exp):
            line = line.strip()
            if "epoch" in line and "lr=" in line and "bs=" in line and "ep=" in line:
                imp_lines.append(line)
        li = 0
        for task in ["mnli", "cola", "mrpc", "sst2"]:
            cur_task_scores = []
            for lr in ["5e-05", "3e-05", "2e-05"]:
                for bsz in ["16", "32"]:
                    for epoch in ["2", "3", "4"]:
                        for j in range(int(epoch)):
                            for i in range(8): # gpus
                                print(li)
                                pattern = "lr=%s, bs=%d, ep=%s"%(lr, int(bsz)//8, epoch)
                                print(imp_lines[li], pattern)
                                assert(pattern in imp_lines[li])
                                cur_task_scores.append(imp_lines[li])
                                li += 1
            print(len(cur_task_scores))
        for line in open(exp):
            line = line.strip()
            print(line)
        sys.exit(0)
    '''
    import codecs
    for exp in glob.glob(master_folder + "/*/sys.err"):
        best_accuracies = {"mnli": 0, "sst2": 0, "cola": 0, "mrpc": 0}
        last_acc, last_task = None, None
        for line in codecs.open(exp, errors="ignore", encoding = "ISO-8859-1"):
            if "/data/home/ganayu/.cache/huggingface/datasets/glue" in line:
                if last_acc:
                    if best_accuracies[last_task] < last_acc:
                        best_accuracies[last_task] = last_acc
                last_task = line.split("/data/home/ganayu/.cache/huggingface/datasets/glue/")[1].split("/")[0]
                last_acc = None
            if "lr=" in line and "bs=" in line:
                if "accuracy" in line:
                    if "'f1'" in line:
                        last_acc = 100.0*float(line.split("'accuracy': ")[1].split(",")[0])
                    else:
                        last_acc = 100.0*float(line.split("'accuracy': ")[1].split("}")[0])
                elif "matthews_correlation" in line:
                    last_acc = 100.0*float(line.split("'matthews_correlation': ")[1].split("}")[0])
        if last_acc:
            if best_accuracies[last_task] < last_acc:
                best_accuracies[last_task] = last_acc
        print(exp, best_accuracies)
# get_best_ft_results_for_more_tasks("/fsx/ganayu/experiments/supershaper/aug2_directfinetune_mnli_supershaper_2xtrainbudget_sandwch_2random")

def get_best_ft_sweep_space():
    target_model = "supershaper"
    for fold in glob.glob("/fsx/ganayu/experiments/supershaper/aug2_supershaper_directfinetune_sweepcheck_*"):
        best_dev_scores = {"cola": 0, "mrpc": 0}
        prev_epoch, prev_score, prev_task = None, None, None
        for l in open(fold + "/" + target_model + "/sys.out"):
            l = l.strip()
            if "epoch" in l and "lr=" in l and "bs=" in l and "ep=" in l:
                epoch = int(l.split("epoch ")[-1].split(":")[0])
                if epoch == 0:
                    if prev_epoch is not None and prev_score > best_dev_scores[prev_task]:
                        best_dev_scores[prev_task] = prev_score
                    prev_epoch = epoch
                    if "f1" in l:
                        prev_score = float(l.split("'f1': ")[1].split("}")[0])
                        prev_task = "mrpc"
                    elif "matthews_correlation" in l:
                        prev_score = float(l.split("'matthews_correlation': ")[1].split("}")[0])
                        prev_task = "cola"
                else:
                    prev_epoch = epoch
                    if "f1" in l:
                        prev_score = float(l.split("'f1': ")[1].split("}")[0])
                        prev_task = "mrpc"
                    elif "matthews_correlation" in l:
                        prev_score = float(l.split("'matthews_correlation': ")[1].split("}")[0])
                        prev_task = "cola"

        if prev_epoch and prev_score > best_dev_scores[prev_task]:
            best_dev_scores[prev_task] = prev_score
        print(fold.split("/")[-1], best_dev_scores)
# get_best_ft_sweep_space()

def get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3(folder, supernet_finetune=False, num_gpus=8, new_supernet_finetune=False):
    experiments = []
    for line in open("/fsx/ganayu/experiments/supershaper/" + folder + "/shell.sh"):
        line = line.strip()
        if "accelerate launch" in line:
            epochs = int(line.split("--num_train_epochs ")[1].split(" ")[0])
            task = line.split("--task_name ")[1].split(" ")[0]
            experiments.append([epochs, task])

    finalscores = {}
    if new_supernet_finetune:
        task2scores = {}
        for line in open("/fsx/ganayu/experiments/supershaper/" + folder + "/sys.out"):
            line = line.strip()      
            if "Val Accuracy" in line and "task" in line and "num_train_epochs" in line:
                result = eval(line[2:])
                if result['epoch'] == result['num_train_epochs']-1:
                    result['SuperTransformer Val Accuracy'] = result['SuperTransformer Val Accuracy']*100.0
                    if result['task'] not in task2scores:
                        task2scores[result['task']] = result['SuperTransformer Val Accuracy']
                    elif task2scores[result['task']] <  result['SuperTransformer Val Accuracy']:
                        task2scores[result['task']] = result['SuperTransformer Val Accuracy']
        finalscores = task2scores
    elif not supernet_finetune:
        cur_exp = -1
        prev_epoch = -1
        task2runs = {}
        last_score = None
        cur_task = None
        for line in open("/fsx/ganayu/experiments/supershaper/" + folder + "/sys.out"):
            line = line.strip()
            if "lr=" in line and "bs=" in line and "ep=" in line:
                if "epoch 0" in line and prev_epoch != 0:
                    if prev_epoch != -1:
                        assert(cur_task!=None)
                        assert(last_score!=None)
                        if cur_task not in task2runs:
                            task2runs[cur_task] = []
                        task2runs[cur_task].append(last_score)
                    cur_exp += 1
                    cur_task = experiments[cur_exp][1]
                prev_epoch = int(line.split("epoch ")[1].split(":")[0])
                last_score = line
        if cur_task not in task2runs:
            task2runs[cur_task] = []
        task2runs[cur_task].append(last_score)

        metric = {"mnli": "accuracy", "cola": "matthews_correlation", "mrpc": "accuracy", "sst2": "accuracy", "qnli": "accuracy", "qqp": "accuracy", "rte": "accuracy", "stsb": "spearmanr"}

        for task in task2runs:
            best_score = None
            best_config = None
            for score in task2runs[task]:
                if score is None:
                    continue
                cur_score = 100.0*float(score.split("'"+metric[task]+"': ")[1].split()[0][0:-1])
                if best_score is None or cur_score > best_score:
                    best_score = cur_score
                    best_config = score
            print(task, best_score)
            finalscores[task] = best_score
    else:
        scores = []
        for line in open("/fsx/ganayu/experiments/supershaper/" + folder + "/sys.out"):
            line = line.strip()
            if "SuperTransformer Val Accuracy" in line:
                scores.append(float(line.split("'SuperTransformer Val Accuracy'")[1].split(":")[1][0:-1]))
        print(scores)
        print(len(scores))
        task2score = {}
        incomplete_tasks = {}
        cur_idx = 0
        final_scores = []
        for exp in experiments:
            epoch, task = exp
            if task not in task2score:
                task2score[task] = None
            if cur_idx+epoch > len(scores):
                incomplete_tasks[task] = True
                break
            cur_score = scores[cur_idx+epoch-1]
            final_scores.append((cur_score, epoch))
            if task2score[task] is None or cur_score > task2score[task]:
                task2score[task] = cur_score
            cur_idx += epoch
        # print(final_scores)
        print(task2score)
        print("incomplete", incomplete_tasks)
        finalscores = {task: 100.0*task2score[task] for task in task2score if task2score[task] is not None}
    res = ""
    for task in ["mnli", "cola", "mrpc", "sst2", "qnli", "qqp", "rte"]: # , "stsb"]:
        if task in finalscores and finalscores[task] is not None:
            res+= "%.2f,"%(finalscores[task])
        else:
            res += ","
    print(res)

# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug14_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_supernet_continue_otherglue/acav1")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug14_directfinetune_v3_v4.1-3_mnli_cola_mrpc_sst2/dftv3")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug14_directfinetune_v3_v4.1-3_mnli_cola_mrpc_sst2/dftv4.1")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug14_directfinetune_v4.2_mnli_cola_mrpc_sst2/dftv4.2")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug19_directfinetune_v4.5_mnli_cola_mrpc_sst2/dftv4.5")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug21_directfinetune_atthard_mnli_cola_mrpc_sst2/dftatthard")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug21_directfinetune_v1_aca_60M_mnli_cola_mrpc_sst2/df60M")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug27_finetune_aca_v3rsub_1e5_mnli_cola_qnli_qqp/acav3")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug28_finetune_aca_v3rsub_4e5_mnli_cola_qnli_qqp/acav34e5")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug28_finetune_aca_v3rsub_3e5_mnli_cola_qnli_qqp/acav33e5")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug27_finetune_aca_v1_supernet_inplace_kd_loghard/loghard")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug27_finetune_aca_v1_supernet_inplace_kd_hid1-12/hid1-12")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_aca_v1_supernet_inplace_kd_hid1-12_nonmnli_8tasks/hid1-12")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_retrain_scratch_nonmnlicola_tasks/v1scratch")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_noretrain_125Ksteps_nonmnlicola_tasks/v1scratch")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_moe2e1rand_125Ksteps_mnli/moe2e1rand")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_inplkd_googlebert_loghard_125Ksteps_mnli/gooft")
# for task in ["cola", "mnli", "sst2", "qnli", "qqp"]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_directfinetune_v1scratch_%s_hardlogits/hl%s"%(task, task), supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_retrain_scratch_nonmnlicola_tasks/v1scratch")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_noretrain_125Ksteps_nonmnlicola_tasks/v1scratch")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_aca_v1_supernet_inplace_kd_hid1-12_nonmnli_8tasks/hid1-12")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug27_finetune_aca_v1_supernet_inplace_kd_hid12/hid12")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug27_finetune_aca_v1_supernet_inplace_kd_hid24681012/hid24681012")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_moe2e1rand_125Ksteps_mnli/moe2e1rand")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_moe4e1rand_125Ksteps_mnli/moe4e1rand")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug30_finetune_acadbertdata_supernet_moe2e1rand_allexpavg_mnli/allexpavg")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug30_finetune_acadbertdata_supernet_moe4e1rand_allexpavg_mnli/allexpavg")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_finetune_acadbertdata_supernet_moe4e1rand_125Ksteps_nonmnlitasks/moe4e1rand")
#for nexp, popsize in [(2, 1), (4, 1), (8, 1)]:
    # get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep4_finetune_acadbertdata_supernet_v1_moe_%de_%drand/%de_%drand"%(nexp, popsize, nexp, popsize))
#    get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep4_finetune_acadbertdata_supernet_v1_moe_%de_%drand_allexpavg/%de_%drand"%(nexp, popsize, nexp, popsize))
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_directfinetune_v3rsub_mnli_hardlogits/31hlmnli", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_directfinetune_v3rsub_cola_hardlogits/31hlcola", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_finetune_v1_subnetkd_logitshard_fthard_mnli_cola_mrpc/31log")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_finetune_v1_subnetkd_logitshid1-12_fthard_mnli_cola_mrpc/31hid1-12")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_finetune_v1_subnetkd_logitshid24681012_fthard_mnli_cola_mrpc/31hid24681012")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_finetune_acadbertdata_supernet_moe2e2rand_125Ksteps_alltasks/2e_2rand")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug31_finetune_acadbertdata_supernet_moe4e2rand_125Ksteps_alltasks/4e_2rand")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep6_finetune_supernetbase_moe_averagingexpert_3e_1rand_allexpavg/ft3e_aleavg")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep6_finetune_supernetbase_moe_averagingexpert_3e_1rand_firstexp/ft3e_1stexp")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep6_finetune_supernetbase_inplacekd_fixedteacher_loghard_hid1-12/ikd1-12")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep6_finetune_acadbertdata_supernet_v1_moe_minmaxrand_corrected_2e_1rand/ft2e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep9_finetune_acadbertdata_supernet_v1_moe_8e_minmaxrand_corrected/ft2e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep9_finetune_acadbertdata_supernet_v1_moe_2e_8e_initialize_other_experts_no/ft2e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep9_finetune_acadbertdata_supernet_v1_moe_minmaxrand_collapsedtraining/ft2e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep9_acadbertdata_supernet_v1_moe_minmaxrand_collapsedtraining_partialcollapsing/ft2e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep11_finetune_acadbertdata_supernet_v1_moe_minmaxrand_corrected_4e/11-4e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep11_finetune_acadbertdata_supernet_v1_moe_minmaxrand_corrected_6e/11-6e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep10_finetune_acadbertdata_supernet_v1_moe_minmaxrand_corrected_2e_1rand_useexpert2/ft2e")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep12_finetune_acadbertdata_supernet_v1_moe_2e_consistencyloss_for_max/12-clmax")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep12_finetune_acadbertdata_supernet_v1_inplacekd_logits_moe2e/12-ikd")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep10_finetune_v1_subnetkdlogits_finetunekdlogits_67M/10s-f-kd", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep12_finetune_subnethardlog_evosearch_supernet_v1_moe2e_67M/12-fsubmoe")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep14_finetune_acadbertdata_supernet_v1_inplacekd_logits_moe2e_consistencyloss_for_max/14-ikdconst")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep14_finetune_v1_subnet_finetunekdlogits_60M/14-60Mft", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep14_directfinetune_v1scratch_mnli_hardlogits_saveckpt/sackmnli", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep15_finetune_v1_subnet_finetunekdlogits_60M_14evoit/15-60Mft", supernet_finetune=True)
# for it in [14, 29]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep15_finetune_v1_subnet_finetunekdlogits_67M_moe2e_%devoit/15moe%dft"%(it,it), supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep18_finetune_v1_subnet_finetunekdlogits_67M_14evoit/18-60Mft", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep19_finetune_v1_subnet_finetunekdlogits_67M_moe/19-67Mmoeft", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep19_finetune_supernet_v1_moe_2xtrainbudget/19-moe250K")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep19_finetune_supernet_v1_sandwich_2xtrainbudget/19-sand250K")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep19_finetune_supernet_v1_moe_seed333/19-sdmoe")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep19_finetune_supernet_v1_moe_seed333_useexpert2/19-2ndexp")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_moe_2e_kd/23-2ekd", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_supernet_kd/23-kd", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_scratch_kd/23-sckd", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_v1_moe_archrouting_1L/archrouting_1L")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_v1_moe_archrouting_2L/archrouting_2L")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct16_finetune_v1_moe_archrouting_jack_1L/ftjack1L")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct16_finetune_v1_moe_archrouting_jack_2L/ftjack2L")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_v1_moe_archrouting_1L_setreqsgrad/archrouting_1L")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct17_finetune_rand2moe_ffncropping_evosearch/fcrps1")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct17_finetune_rand2moe_ffncropping_evosearch_seed333/fcrps2")

# slide 114
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct20_finetune_moe_archrouting_jack_2L_64/2L_64")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct20_finetune_moe_archrouting_jack_2L_128/2L_128")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct20_finetune_moe_archrouting_jack_2L_seed333/2L_64")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct20_finetune_moe_archrouting_jack_2L_fixedarch_64/2L_64")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct20_finetune_moe_archrouting_jack_2L_fixedarch_128/2L_128")

# different finetuning seed
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_retrain_subnet_125Ksteps_retrain_scratch_nonmnlicola_tasks/v1scratch") # scratch
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("aug29_finetune_acadbertdata_supernet_noretrain_125Ksteps_nonmnlicola_tasks/v1scratch") # supernet
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct21_finetune_moe_archrouting_jack_2L_128_seed333/2L_128")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct21_finetune_scratch_seed333/scratch")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct21_finetune_supernet_seed333/supernet")

# different num_experts
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct21_finetune_moe_archrouting_jack_2L_moreexperts_exp4/exp4")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct21_finetune_moe_archrouting_jack_2L_moreexperts_exp6/exp6")

# seed 444
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct27_finetune_scratch_seed444/scratch")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct27_finetune_supernet_seed444/supernet")
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct27_finetune_moe_archrouting_jack_2L_128_seed444/2L_128")

# kd ft
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_scratch_kd/23-sckd", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_supernet_kd/23-kd", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_moe_2e_kd/23-2ekd", supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct28_finetune_archmoe_pavg_2e_kd_seed333/28-archkd", supernet_finetune=True)

# v1.1 ft
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep25_finetune_supernet_v1p1_67M/25-1p16-7M", supernet_finetune=True)

# kd_corrected
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_scratch_kd_corrected/23-sckd", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_supernet_kd_corrected/23-kd", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct28_finetune_archmoe_pavg_2e_kd_seed333_corrected/28-archkd", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct28_finetune_archpavg_2e_onehot_kd_seed333/28-onehot", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct28_finetune_archpavg_2e_onehot_kd_seed333_part2/28-onehot", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_scratch_kd_corrected_part2/scratch", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_supernet_kd_corrected_part2/supernet", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct28_finetune_archmoe_pavg_2e_kd_seed333_corrected_part2/archmoe", new_supernet_finetune=True)
# neuronrouting_jack_2L
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov12_finetune_neuron_2L_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)
# neuronrouting_jack_2L
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov12_finetune_standalone_67M_3xbudget_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# 50M
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("nov6_finetune_archmoe_50M/6-p1", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("nov6_finetune_archmoe_50M_part2/6-p2", new_supernet_finetune=True)

# seed analysis on small datasets
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_scratch_kd_corrected_cola_seed444/23-sckd", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_scratch_kd_corrected_part2_mrpc_rte_seed444/scratch", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_supernet_kd_corrected_cola_seed444/23-kd", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("sep23_finetune_supernet_kd_corrected_part2_mrpc_rte_seed444/supernet", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct28_finetune_archmoe_pavg_2e_kd_seed333_corrected_cola_seed444/28-archkd", new_supernet_finetune=True)
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("oct28_finetune_archmoe_pavg_2e_kd_seed333_corrected_part2_mrpc_rte_seed444/archmoe", new_supernet_finetune=True)

# mnli as proxy task (archmoe)
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#   get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("nov16_finetune_evosearch_archmoe_supernet_proxy_mnli_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)

# comparison to nas-bert
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("nov16_finetune_standalone_archmoe_60M_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)

# collapse_experts_before_ft==1
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("nov19_collapse_and_finetune_neuronmoe_2L_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("nov19_collapse_and_finetune_archmoe_2L_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)

# collapse and finetune
# neuronrouting_jack_2L
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov19_collapse_and_finetune_neuronmoe_2L_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)
# arch. moe
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
# get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov19_collapse_and_finetune_archmoe_2L_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)
# autodistill - 50M
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov20_collapse_and_finetune_archmoe_50M_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)
# nasbert - 60M
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov20_collapse_and_finetune_standalone_archmoe_60M_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)

# comparison to autodistill
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov19_finetune_archmoe_50M_%s_largerepochs/%s"%(exp_name, exp_name), new_supernet_finetune=True)
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov22_finetune_neuronmoe_50M_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# larger lr 
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov26_finetune_neuronmoe_67M_%s_lr8e-4/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# neuronrouting_jack_drop_2L
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov26_finetune_neuronmoe_drop_67M_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# nas-bert comparison 67M (neuron_moe)
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("nov21_finetune_standalone_neuronmoe_60M_%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)

# comparison to autodistill - 50M
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/nov22_finetune_neuronmoe_50M_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# v1.2 - supernet training -- ffn elastic
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec6_finetune_neuronmoe_27M_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# neuron jack drop 2L - dropout layer removed
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec7_finetune_neuronmoe_drop_corrected_67M_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# neuron jack 2L - autodistil - 27M
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec8_finetune_neuronmoe_corrected_27M_%s/%s"%(exp_name, exp_name), new_supernet_finetune=True)

# neuron jack 2L - autodistil - 50M
# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec8_finetune_neuronmoe_50M_%s_lr8e-4/%s"%(exp_name, exp_name), new_supernet_finetune=True)

'''
# neuron jack 2L - "5M", "10M", "27M", "50M"
for model_size in ["5M", "27M", "10M", "50M"]:
  print(model_size)
  for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
    get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec19_finetune_neuronmoe_v5/%s_%s/%s"%(model_size, exp_name, exp_name), new_supernet_finetune=True)
'''

'''
for model_size in ["30M", "60M", "67M"]:
  print(model_size)
  for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
    get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec21_finetune_neuronmoe_v5/%s_%s/%s"%(model_size, exp_name, exp_name), new_supernet_finetune=True)
'''

'''
for model_size in ["27M",  "10M"]:
  print(model_size)
  for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
    get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec24_finetune_neuronmoe_v5p2/%s_%s/%s"%(model_size, exp_name, exp_name), new_supernet_finetune=True)
'''

# for model_size in ["27M",  "50M"]:
#  print(model_size)
#  for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#    get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/dec26_finetune_neuronmoe_v5p1_250Ksteps/%s_%s/%s"%(model_size, exp_name, exp_name), new_supernet_finetune=True)

# for exp_name, tasks in [("mnli_mrpc_rte", ["mnli", "mrpc", "rte"]), ("cola_qqp", ["cola", "qqp"]), ("sst2_qnli", ["sst2", "qnli"])]:
#  get_scores_for_create_finetuning_experiments_standalone_vs_supernet_v3("/jan3_finetune_archmoe_jack_2L_fixedarch/%s/%s"%(exp_name,exp_name), new_supernet_finetune=True)


def get_pareto_curve(plot_output=None, iteration=None, experiments=None, sheet_name=None):
    os.makedirs(plot_output, exist_ok=True)

    from xlrd import open_workbook
    pareto_scores = {}
    for exp in experiments:
        for f in glob.glob("/fsx/ganayu/experiments/supershaper/%s/*/%s"%(exp, iteration)):
            rb = open_workbook(f, formatting_info=True)
            best_config_sheet = rb.sheet_by_name(sheet_name)
            model_name = f.split("/")[-2]
            pareto_scores[model_name] = [[], []]
            ri = 1
            while True:
                try:
                    cur_model_size = best_config_sheet.cell(ri, 2).value
                except:
                    break
                cur_model_size = int(cur_model_size)
                cur_valid_ppl = float(best_config_sheet.cell(ri, 3).value)
                pareto_scores[model_name][0].append(cur_model_size)
                pareto_scores[model_name][1].append(cur_valid_ppl)
                ri += 1
    print("loaded %d experiments"%(len(pareto_scores)))

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(13,7))
    colors = ["violet", 'r', 'b', "springgreen", "gold", 'c', 'y', "indigo", 'm',  'g', 'k', "olive", "firebrick"]
    ei = 0
    for model in sorted(pareto_scores):
        sns.scatterplot(x=pareto_scores[model][0], y=pareto_scores[model][1], color=colors[ei], label=model)
        ei += 1
    plt.xlabel("Model Size", fontsize=15)
    plt.ylabel("Valid. PPL", fontsize=15)
    plt.legend(loc="upper left", ncol=2, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fig.savefig("%s/pareto-curve.png"%(plot_output))

# get_pareto_curve(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug3_pareto_diff_search_spaces", iteration="evo_results_29.xls", sheet_name="iter_10", experiments=["aug1_v3_supernetbase_search_different_spaces"])
# get_pareto_curve(plot_output="/fsx/ganayu/experiments/supershaper/summary_plots/aug1_v3_supernetbase_search_diff_configs", iteration="evo_results_29.xls", sheet_name="iter_10", experiments=["aug1_v3_supernetbase_search_diff_configs"])
# def collapse_sparse_to_dense_model_using_paramaverage(src_ckpt, dest_ckpt):
# collapse_sparse_to_dense_model_using_paramaverage("/fsx/ganayu/experiments/supershaper/aug27_acadbertdata_supernet_v1_moe/2e_1rand/best_model", "/fsx/ganayu/experiments/supershaper/aug27_acadbertdata_supernet_v1_moe/2e_1rand/all_average_model")




