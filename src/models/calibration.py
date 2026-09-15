import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict

def calculate_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> float:
        temp_param = nn.Parameter(torch.ones(1) * 1.5)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([temp_param], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(logits / temp_param, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        self.temperature = max(0.1, temp_param.item())
        return self.temperature
