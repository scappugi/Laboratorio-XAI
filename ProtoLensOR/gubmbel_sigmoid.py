"""
Gumbel_Sigmoid.py

Modulo mancante nel repository pubblico di ProtoLens: AdaptiveMask.py fa
`from Gumbel_Sigmoid import *` e PLens.py istanzia `GumbelSigmoid()`, ma il
file non è incluso nel push su GitHub.

Questa è un'implementazione standard del trucco di reparametrizzazione
Gumbel-Sigmoid (variante binaria del Gumbel-Softmax / Concrete
distribution), usata per ottenere una maschera binaria differenziabile a
partire da logit continui, con uno stimatore straight-through.

Compatibile con la firma attesa in AdaptiveMask.get_mask():
    self.gumbel(attention_logits, Log=log, mask=mask, return_soft=return_soft)
    self.gumbel(attention_logits, mask=mask, return_soft=return_soft)
"""

import torch
import torch.nn as nn


class GumbelSigmoid(nn.Module):
    def __init__(self, temperature=1.0, eps=1e-10):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def _sample_gumbel(self, shape, device):
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + self.eps) + self.eps)

    def forward(self, logits, Log=False, mask=None, return_soft=False, hard=True):
        if self.training:
            noise = self._sample_gumbel(logits.shape, logits.device)
            y = torch.sigmoid((logits + noise) / self.temperature)
        else:
            y = torch.sigmoid(logits / self.temperature)

        if mask is not None:
            y = y * mask

        if return_soft:
            return y

        if hard:
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y  # straight-through estimator

        if Log:
            print(f"[GumbelSigmoid] mean activation: {y.mean().item():.4f}")

        return y
