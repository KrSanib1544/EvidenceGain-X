from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import torch

@dataclass
class HypothesisRecord:
    disease: str
    class_idx: int
    probability: float
    role: str # 'leading', 'critical_rival', 'secondary', 'low_prob'

class CompetingHypothesisTracker:
    def __init__(self, classes: List[str], top_k: int = 3, margin_threshold: float = 0.25):
        self.classes = classes
        self.top_k = min(top_k, len(classes))
        self.margin_threshold = margin_threshold

    def analyze(self, probs: torch.Tensor) -> Dict[str, Any]:
        if probs.dim() == 2:
            probs = probs.squeeze(0)
        
        p_np = probs.detach().cpu().numpy()
        sorted_indices = np.argsort(-p_np)
        
        hypotheses: List[HypothesisRecord] = []
        for rank, idx in enumerate(sorted_indices[:self.top_k]):
            p_val = float(p_np[idx])
            if rank == 0:
                role = "leading"
            elif rank == 1 and (p_np[sorted_indices[0]] - p_val) <= self.margin_threshold:
                role = "critical_rival"
            elif rank == 1:
                role = "secondary"
            else:
                role = "low_prob"
            
            hypotheses.append(HypothesisRecord(
                disease=self.classes[idx],
                class_idx=int(idx),
                probability=p_val,
                role=role
            ))

        margin = float(p_np[sorted_indices[0]] - p_np[sorted_indices[1]])
        eps = 1e-8
        entropy = float(-np.sum(p_np * np.log2(p_np + eps)))
        is_ambiguous = margin < self.margin_threshold

        return {
            "leading_disease": self.classes[sorted_indices[0]],
            "leading_prob": float(p_np[sorted_indices[0]]),
            "rival_disease": self.classes[sorted_indices[1]] if len(sorted_indices) > 1 else None,
            "rival_prob": float(p_np[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0,
            "margin": margin,
            "entropy": entropy,
            "is_ambiguous": is_ambiguous,
            "top_hypotheses": hypotheses,
            "full_distribution": {self.classes[i]: float(p_np[i]) for i in range(len(self.classes))}
        }
