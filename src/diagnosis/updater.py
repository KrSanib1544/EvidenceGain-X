from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import numpy as np

class MultimodalDiagnosisUpdater(nn.Module):
    def __init__(self, embed_dim: int = 512, num_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Evidence projection for image / tabular features
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        base_features: torch.Tensor,
        evidence_features: torch.Tensor,
        quality_score: float,
        contradiction_penalty: float
    ) -> torch.Tensor:
        # base_features: (B, embed_dim), evidence_features: (B, embed_dim)
        b_size = base_features.size(0)
        q_tensor = torch.full((b_size, 1), quality_score, dtype=torch.float32, device=base_features.device)
        p_tensor = torch.full((b_size, 1), contradiction_penalty, dtype=torch.float32, device=base_features.device)

        # Modulate evidence by reliability quality
        weighted_ev = evidence_features * quality_score
        
        fused_input = torch.cat([base_features, weighted_ev, q_tensor, p_tensor], dim=-1)
        delta_logits = self.fusion_gate(fused_input)
        return delta_logits

class RuleBasedBeliefUpdater:
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.num_classes = len(classes)

    def update_beliefs(
        self,
        prior_probs: np.ndarray,
        evidence_affinity: np.ndarray,
        quality_score: float,
        contradiction_penalty: float
    ) -> np.ndarray:
        # Bayesian likelihood weighting: P(D|X_0, e) ∝ P(D|X_0) * (L(e|D) ^ (quality * penalty))
        eff_weight = quality_score * contradiction_penalty
        likelihood = np.power(np.maximum(1e-4, evidence_affinity), eff_weight)
        posterior = prior_probs * likelihood
        posterior_sum = posterior.sum()
        if posterior_sum > 0:
            posterior /= posterior_sum
        else:
            posterior = prior_probs.copy()
        return posterior
