# Code is modified from MEAL (https://arxiv.org/abs/1812.02425) and Label Refinery (https://arxiv.org/abs/1805.02641).

import torch
from torch.nn import functional as F
from torch.nn.modules import loss


#class DistributionLoss(loss._Loss):
"""The KL-Divergence loss for the binary student model and real teacher output.

output must be a pair of (model_output, real_output), both NxC tensors.
The rows of real_output must all add up to one (probability scores);
however, model_output must be the pre-softmax output of the network."""

def summer_forward(model_output, real_output):

    size_average = True

    # Target is ignored at training time. Loss is defined as KL divergence
    # between the model output and the refined labels.
    if real_output.requires_grad:
        raise ValueError("real network output should not require gradients.")

    model_output_log_prob = F.log_softmax(model_output, dim=1)
    real_output_soft = F.softmax(real_output, dim=1)
    del model_output, real_output

    # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
    # for batch matrix multiplicatio
    real_output_soft = real_output_soft.unsqueeze(1)
    model_output_log_prob = model_output_log_prob.unsqueeze(2)

    # Compute the loss, and average/sum for the batch.
    cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
    if size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
    else:
            cross_entropy_loss = cross_entropy_loss.sum()
    # Return a pair of (loss_output, model_output). Model output will be
    # used for top-1 and top-5 evaluation.
    # model_output_log_prob = model_output_log_prob.squeeze(2)
    return cross_entropy_loss

#  https://github.com/pytorch/pytorch/issues/11959
#class CrossEntropyLossSoft(loss):
def supershaper_forward(preds, target_logits, reduction="mean"):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(
        preds.view(preds.shape[0], -1), dim=1
    )
    target = torch.nn.functional.softmax(
        target_logits.view(target_logits.shape[0], -1).detach(), dim=1
    )
    batchloss = -torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == "none":
        return batchloss
    elif reduction == "mean":
        return torch.mean(batchloss)
    elif reduction == "sum":
        return torch.sum(batchloss)
    else:
        raise NotImplementedError("Unsupported reduction mode.")

def test():
    # for i in range(10):
    pred = torch.Tensor([[1, 2.2, 3]])
    gold = torch.Tensor([[1, 2, 3.1]])
    print(pred)
    sup_score = supershaper_forward(pred, gold)
    sum_score = summer_forward(pred, gold)
    print(sup_score, sum_score)

#test()