"""Answer-slot projection and same-size distribution distillation loss."""
import torch
import torch.nn.functional as F


class Readout(torch.nn.Module):
    def __init__(self, base, label_ids, full_rank):
        super().__init__()
        self.base = base
        self.full_rank = full_rank
        self.register_buffer('label_ids', torch.tensor(label_ids), persistent=False)
        if not full_rank:
            weight = base.get_base_model().get_output_embeddings().weight[label_ids].detach().clone()
            self.register_buffer('legal_weight', weight, persistent=False)

    def forward(self, input_ids, attention_mask):
        base = self.base if self.full_rank else self.base.get_base_model()
        backbone = getattr(base.model, 'language_model', base.model)
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=True).last_hidden_state
        slot = hidden[torch.arange(len(hidden), device=hidden.device), attention_mask.sum(-1)-1]
        weight = (base.get_output_embeddings().weight.index_select(0, self.label_ids)
                  if self.full_rank else self.legal_weight)
        return F.linear(slot, weight).float()


def loss_terms(logits, rows, weights):
    lengths = torch.tensor([len(row['options']) for row in rows], device=logits.device)
    masked = logits.masked_fill(torch.arange(logits.shape[-1], device=logits.device)[None, :]
                                >= lengths[:, None], -1e9)
    logs = masked.log_softmax(-1)
    teacher = torch.zeros_like(logits)
    baseline = torch.zeros_like(logits)
    for i, row in enumerate(rows):
        teacher[i, :lengths[i]] = torch.tensor(row['train_target'], device=logits.device)
        baseline[i, :lengths[i]] = torch.tensor(row['p0'], device=logits.device)
    weight = torch.tensor([weights[row['type']] for row in rows], device=logits.device)
    ce = weight * -(teacher * logs).sum(-1)
    ordinal = weight * .5 * torch.tensor([row['type'] == 'score' for row in rows], device=logits.device)
    ordinal = ordinal * ((logs.exp()-teacher) * torch.arange(logits.shape[-1], device=logits.device)).sum(-1).square()
    ordinal = ordinal / (lengths-1).square()
    kappa = torch.tensor([.2 if row['type'] == 'noul' and max(row['gold_target']) >= 1-1e-6
                          else .05 for row in rows], device=logits.device)
    kl = kappa * (baseline * (baseline.clamp_min(1e-30).log()-logs)).sum(-1)
    return ce + ordinal + kl, ce, ordinal, kl
