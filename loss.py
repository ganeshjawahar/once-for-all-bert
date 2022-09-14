import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
# from custom_layers.KD_loss import DistributionLoss 
from torch.nn import functional as F

def compute_layerwise_distillation(
    args,
    teacher_hidden_states,
    student_hidden_states,
    teacher_attention_maps,
    student_attention_maps,
    track_layerwise_loss=False,
):

    student_fkt = 0.0
    student_akt = 0.0
    if track_layerwise_loss:
        layer_wise_fkt = []
        layer_wise_akt = []
    else:
        layer_wise_fkt = None
        layer_wise_akt = None

    non_trainable_layernorm = nn.LayerNorm(
        teacher_hidden_states[-1].shape[1:], elementwise_affine=False
    )
    for teacher_hidden, student_hidden in zip(
        teacher_hidden_states[-1:], student_hidden_states[-1:]
    ):
        teacher_hidden = non_trainable_layernorm(teacher_hidden.detach()) 
        student_hidden = non_trainable_layernorm(student_hidden)
        fkt = nn.MSELoss()(teacher_hidden, student_hidden)
        student_fkt = student_fkt + fkt
        if track_layerwise_loss:
            layer_wise_fkt.append(fkt)

    # the attention maps already have softmax applied, hence we pass logits = False
    loss_alpha_div = AdaptiveLossSoft(
        args.alpha_min, args.alpha_max, args.beta_clip, logits=False
    )
    
    for (teacher_attention, student_attention) in zip(
        teacher_attention_maps[-1:], student_attention_maps[-1:]
    ):
        # attentions are already in probabilities, hence no softmax
        if args.alpha_divergence:
            # TODO - Check if the reduction is mean or sum
            student_akt = loss_alpha_div(teacher_attention, student_attention)
        else:
            student_attention = student_attention.clamp(min=1e-4).log()
            student_kl = -(teacher_attention.detach() * student_attention)
            akt = torch.mean(torch.sum(student_kl, axis=-1))
            student_akt = student_akt + akt

        if track_layerwise_loss:
            layer_wise_akt.append(akt)

    return student_akt, student_fkt, layer_wise_akt, layer_wise_fkt


def compute_student_loss(
    outputs,
    teacher_info,
    args,
    track_layerwise_loss=False,
    logits_kd=False
):

    # outputs = model(**batch, use_soft_loss=True)
    loss = outputs.loss
    # student_hidden_states = outputs.hidden_states
    # student_attention_maps = outputs.attentions

    student_mlm_loss = loss
    student_mlm_loss = student_mlm_loss / args.gradient_accumulation_steps

    overall_loss = 0.0
    losses = {}
    losses["student_mlm_loss"] = student_mlm_loss.item()
    
    if (args.distillation_type is not None and "logits" in args.distillation_type) or logits_kd:
        # compute KL-div of student logits and teacher logits
        # https://raw.githubusercontent.com/liuzechun/ReActNet/master/utils/KD_loss.py
        student_logits = outputs.logits
        if student_logits.dim() == 3:
            student_logits = student_logits.reshape(-1, student_logits.size(2))
        model_output_log_prob = F.log_softmax(student_logits, dim=1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        teacher_logits = teacher_info["teacher_logits"]
        real_output_soft = F.softmax(teacher_logits.reshape(-1, teacher_logits.size(2)) if teacher_logits.dim() == 3 else teacher_logits, dim=1)
        real_output_soft = real_output_soft.unsqueeze(1)

        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        # overall_loss = cross_entropy_loss * args.inplace_kd_distill_loss_weights
        student_distill_loss = 0.0
        if (args.distillation_type is not None and "hard" in args.distillation_type) and not logits_kd:
            student_distill_loss = args.inplace_kd_distill_loss_contrib * student_mlm_loss + args.inplace_kd_hard_loss_contrib * cross_entropy_loss
        else:
            student_distill_loss = cross_entropy_loss
        student_distill_loss = student_distill_loss / args.gradient_accumulation_steps
        overall_loss = overall_loss + student_distill_loss
        losses["student_distill_loss"] = cross_entropy_loss.item()
    
    if (args.distillation_type is not None and "hiddenlastlayer" in args.distillation_type):
        fkt = 0.0
        num_layers_distilled = 0
        for layer_id in args.inplace_kd_layers:
            if layer_id >= len(teacher_info["teacher_hidden_states"]):
                continue
            non_trainable_layernorm = nn.LayerNorm(teacher_info["teacher_hidden_states"][layer_id].shape[1:], elementwise_affine=False)
            teacher_hidden, student_hidden  = teacher_info["teacher_hidden_states"][layer_id], outputs.hidden_states[layer_id]
            teacher_hidden = non_trainable_layernorm(teacher_hidden.detach()) 
            student_hidden = non_trainable_layernorm(student_hidden)
            cur_fkt = nn.MSELoss()(teacher_hidden, student_hidden)
            fkt = fkt + cur_fkt
            num_layers_distilled += 1
        fkt = fkt / float(num_layers_distilled)

        student_hidden_loss = 0.0
        if "hard" in args.distillation_type:
            student_hidden_loss = args.inplace_kd_distill_loss_contrib * student_mlm_loss + args.inplace_kd_hard_loss_contrib * fkt
        else:
            student_hidden_loss = fkt
        student_hidden_loss = student_hidden_loss / args.gradient_accumulation_steps
        overall_loss = overall_loss + student_hidden_loss
        losses["student_hidden_loss"] = fkt.item()
    
    if (args.distillation_type is not None and "attentionlastlayer" in args.distillation_type):
        akt = 0.0
        num_layers_distilled = 0
        for layer_id in args.inplace_kd_layers:
            if layer_id >= len(teacher_info["teacher_attention_maps"]):
                continue
            teacher_attention, student_attention = teacher_info["teacher_attention_maps"][layer_id], outputs.attentions[layer_id]
            student_attention = student_attention.clamp(min=1e-4).log()
            student_kl = -(teacher_attention.detach() * student_attention)
            cur_akt = torch.mean(torch.sum(student_kl, axis=-1))
            akt = akt + cur_akt
            num_layers_distilled += 1
        akt = akt / float(num_layers_distilled)

        student_attention_loss = 0.0
        if "hard" in args.distillation_type:
            student_attention_loss = args.inplace_kd_distill_loss_contrib * student_mlm_loss + args.inplace_kd_hard_loss_contrib * akt
        else:
            student_attention_loss = akt
        student_attention_loss = student_attention_loss / args.gradient_accumulation_steps
        overall_loss = overall_loss + student_attention_loss
        losses["student_attention_loss"] = akt.item()
    
    if (args.distillation_type is not None and "tinybert" in args.distillation_type):
        # hidden loss
        fkt = 0.0
        num_layers_distilled = 0
        for layer_id in args.inplace_kd_layers:
            if layer_id >= len(teacher_info["teacher_hidden_states"]):
                continue
            teacher_hidden, student_hidden  = teacher_info["teacher_hidden_states"][layer_id], outputs.hidden_states[layer_id]
            cur_fkt = nn.MSELoss()(teacher_hidden.detach(), student_hidden)
            fkt = fkt + cur_fkt
            num_layers_distilled += 1

        '''
        # attention loss
        akt = 0.0
        num_layers_distilled = 0
        for layer_id in args.inplace_kd_layers:
            if layer_id >= len(teacher_info["teacher_attention_maps"]):
                continue
            teacher_attention, student_attention = teacher_info["teacher_attention_maps"][layer_id], outputs.attentions[layer_id]
            teacher_attention.detach()
            student_attention.detach()
            # cur_akt = nn.MSELoss()(teacher_attention.detach(), student_attention)
            # akt = akt + cur_akt
            num_layers_distilled += 1
        '''

        # final distillation loss
        final_distillation_loss = 0.0
        if "hard" in args.distillation_type:
            final_distillation_loss = args.inplace_kd_distill_loss_contrib * student_mlm_loss + args.inplace_kd_hard_loss_contrib * (fkt) # + akt.item())
        else:
            final_distillation_loss = fkt # + akt 

        final_distillation_loss = final_distillation_loss / args.gradient_accumulation_steps
        overall_loss = overall_loss + final_distillation_loss
        losses["student_hidden_loss"] = fkt.item()

    losses["overall_loss"] = overall_loss.item()

    # overall_loss = student_mlm_loss

    '''
    losses = {
        "overall_loss": overall_loss,
        "student_distill_loss": 0,
        "student_mlm_loss": student_mlm_loss,
        "student_feature_knowledge_transfer_loss": 0,
        "student_attention_knowledge_transfer_loss": 0,
        "layer_wise_akt": [],
        "layer_wise_fkt": [],
    }

    if args.layerwise_distillation or args.distillation_type:
        (
            student_akt,
            student_fkt,
            layer_wise_akt,
            layer_wise_fkt,
        ) = compute_layerwise_distillation(
            # the official mobilbeBert repo skips the first layer
            # teacher_hidden_states[1:],
            # student_hidden_states[1:],
            # teacher_attention_maps[1:],
            # student_attention_maps[1:],
            args,
            teacher_hidden_states,
            student_hidden_states,
            teacher_attention_maps,
            student_attention_maps,
            track_layerwise_loss=track_layerwise_loss,
        )

        student_distill_loss = 0.5 * student_fkt + 0.5 * student_akt
        student_distill_loss = student_distill_loss / args.gradient_accumulation_steps

        overall_loss = overall_loss + student_distill_loss

        losses["overall_loss"] = overall_loss
        losses["student_distill_loss"] = student_distill_loss
        losses["student_feature_knowledge_transfer_loss"] = student_fkt
        losses["student_attention_knowledge_transfer_loss"] = student_akt
        losses["layer_wise_akt"] = layer_wise_akt
        losses["layer_wise_fkt"] = layer_wise_fkt
    '''

    return overall_loss, losses


## Alpha Divergence loss codes adapted from https://github.com/facebookresearch/AlphaNet ##
def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3, logits=True):
    assert isinstance(alpha, float)
    if logits:
        q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
        p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
        q_log_prob = torch.nn.functional.log_softmax(
            q_logits, dim=1
        )  # gradient is only backpropagated here
    else:
        q_prob = q_logits.detach()
        p_prob = p_logits.detach()
        p_prob = p_prob.view(p_prob.shape[0], -1)  ### Getting the correct view
        q_log_prob = q_logits.log()

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss


"""
It's often necessary to clip the maximum
gradient value (e.g., 1.0) when using this adaptive KD loss
"""


class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    def __init__(self, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0, logits=True):
        super(AdaptiveLossSoft, self).__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip
        self.logits = logits

    def forward(self, output, target, alpha_min=None, alpha_max=None):
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        loss_left, grad_loss_left = f_divergence(
            output, target, alpha_min, iw_clip=self.iw_clip, logits=self.logits
        )
        loss_right, grad_loss_right = f_divergence(
            output, target, alpha_max, iw_clip=self.iw_clip, logits=self.logits
        )

        ind = torch.gt(loss_left, loss_right).float()
        loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

        # reduction is mean by default https://pytorch-enhance.readthedocs.io/en/latest/_modules/torch/nn/modules/loss.html
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def ce_soft(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


#  https://github.com/pytorch/pytorch/issues/11959
class CrossEntropyLossSoft(_Loss):
    def forward(self, preds, target_logits, reduction="mean"):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        print(preds.size(), target_logits.size())
        logprobs = torch.nn.functional.log_softmax(
            preds.view(preds.shape[0], -1), dim=1
        )
        target = torch.nn.functional.softmax(
            target_logits.view(target_logits.shape[0], -1).detach(), dim=1
        )
        print(target.view(target.shape[0], -1).size(), logprobs.size())
        batchloss = -torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        
        if reduction == "none":
            return batchloss
        elif reduction == "mean":
            return torch.mean(batchloss)
        elif reduction == "sum":
            return torch.sum(batchloss)
        else:
            raise NotImplementedError("Unsupported reduction mode.")

