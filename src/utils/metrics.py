import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_mae_train(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # var = torch.mean(torch.var(loss, dim=0))
    return torch.mean(loss), torch.var(loss)



def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # loss = torch.abs(preds - labels) / labels
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse


def temporal_similarity_soft(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / (labels + 1)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def temporal_similarity_hard(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = - (torch.mul(labels, torch.log(preds))) - (torch.mul((1. - labels), torch.log(1. - preds)))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def temporal_similarity_hard_f1_e(preds, labels, null_val, e=0.1):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    TP = torch.sum((labels - preds) <= e)
    Precision = TP / torch.sum(labels == 1)
    Recall = TP / torch.sum(preds >= 1 - e)
    loss = 2 * (Precision * Recall) / (Precision  + Recall)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def compute_all_similarity(preds, labels, null_val):
    tss = temporal_similarity_soft(preds, labels, null_val).item()
    tsh = temporal_similarity_hard(preds, labels, null_val).item()
    tsh_f1 = temporal_similarity_hard_f1_e(preds, labels, null_val).item()
    return tss, tsh, tsh_f1



def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx, subgraph_idx_base, num_canonical_order, 
                           sum_order_log_prob=False, return_neg_log_prob=False, reduction="mean"):
  num_subgraph = subgraph_idx_base[-1]
  B = subgraph_idx_base.shape[0] - 1
  C = num_canonical_order
  E = log_theta.shape[0]
  K = log_theta.shape[1]
  assert E % C == 0
  adj_loss = torch.stack(
      [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)

  const = torch.zeros(num_subgraph).to(label.device)
  const = const.scatter_add(0, subgraph_idx,
                            torch.ones_like(subgraph_idx).float())

  reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
  reduce_adj_loss = reduce_adj_loss.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)

  reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
  reduce_log_alpha = reduce_log_alpha.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
  reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
  reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

  log_prob = -reduce_adj_loss + reduce_log_alpha
  log_prob = torch.logsumexp(log_prob, dim=1)

  bc_log_prob = torch.zeros([B*C]).to(label.device)
  bc_idx = torch.arange(B*C).to(label.device)
  bc_const = torch.zeros(B*C).to(label.device)
  bc_size = (subgraph_idx_base[1:] - subgraph_idx_base[:-1]) // C
  bc_size = torch.repeat_interleave(bc_size, C)
  bc_idx = torch.repeat_interleave(bc_idx, bc_size)
  bc_log_prob = bc_log_prob.scatter_add(0, bc_idx, log_prob)

  bc_const = bc_const.scatter_add(0, bc_idx, const)
  bc_loss = (bc_log_prob / bc_const)

  bc_log_prob = bc_log_prob.reshape(B,C)
  bc_loss = bc_loss.reshape(B,C)
  if sum_order_log_prob:
    b_log_prob = torch.sum(bc_log_prob, dim=1)
    b_loss = torch.sum(bc_loss, dim=1)
  else:
    b_log_prob = torch.logsumexp(bc_log_prob, dim=1)
    b_loss = torch.logsumexp(bc_loss, dim=1)
      
  b_neg_log_prob = -2*b_log_prob
  b_loss = -b_loss
  
  if reduction == "mean":
    neg_log_prob = b_neg_log_prob.mean()
    loss = b_loss.mean()
  elif reduction == "sum":
    neg_log_prob = b_neg_log_prob.sum()
    loss = b_loss.sum()
  else:
    assert reduction == "none"
    neg_log_prob = b_neg_log_prob
    loss = b_loss
  
  if return_neg_log_prob:
    return loss, neg_log_prob
  else:
    return loss
