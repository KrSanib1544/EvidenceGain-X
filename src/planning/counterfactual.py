from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

class CounterfactualEvidencePlanner:
    def __init__(self, classes: List[str], candidate_types: List[str]):
        self.classes = classes
        self.candidate_types = candidate_types
        self.num_classes = len(classes)

        # Domain discriminative affinity matrix: Prior expectation of how well evidence e separates (disease_i, disease_j)
        # 1.0 = highly discriminative, 0.2 = non-discriminative
        self.affinity_matrix: Dict[str, np.ndarray] = {
            "lesion_closeup": np.array([
                [0.0, 0.9, 0.8, 0.9],  # Early Blight vs Late / Septoria
                [0.9, 0.0, 0.85, 0.9], # Late Blight vs Early / Septoria
                [0.8, 0.85, 0.0, 0.9], # Septoria vs others
                [0.9, 0.9, 0.9, 0.0]
            ], dtype=np.float32),
            "leaf_underside": np.array([
                [0.0, 0.95, 0.4, 0.8], # Late blight shows white mold/sporulation on underside
                [0.95, 0.0, 0.7, 0.8],
                [0.4, 0.7, 0.0, 0.6],
                [0.8, 0.8, 0.6, 0.0]
            ], dtype=np.float32),
            "stem_view": np.array([
                [0.0, 0.6, 0.2, 0.5],
                [0.6, 0.0, 0.3, 0.5],
                [0.2, 0.3, 0.0, 0.3],
                [0.5, 0.5, 0.3, 0.0]
            ], dtype=np.float32),
            "symptom_text": np.array([
                [0.0, 0.8, 0.8, 0.9],
                [0.8, 0.0, 0.7, 0.9],
                [0.8, 0.7, 0.0, 0.9],
                [0.9, 0.9, 0.9, 0.0]
            ], dtype=np.float32),
            "weather_context": np.array([
                [0.0, 0.85, 0.5, 0.3], # Late blight favors cool/wet, early blight favors warm/dew
                [0.85, 0.0, 0.5, 0.3],
                [0.5, 0.5, 0.0, 0.2],
                [0.3, 0.3, 0.2, 0.0]
            ], dtype=np.float32)
        }

    def simulate_counterfactual_outcomes(
        self,
        current_probs: np.ndarray,
        h1_idx: int,
        h2_idx: int,
        evidence_type: str
    ) -> Dict[str, Any]:
        aff_mat = self.affinity_matrix.get(
            evidence_type,
            np.ones((self.num_classes, self.num_classes), dtype=np.float32) * 0.5
        )
        sep_power = float(aff_mat[h1_idx, h2_idx])

        # Counterfactual Outcome A: Evidence supports H1
        p_outcome_a = current_probs.copy()
        shift = p_outcome_a[h2_idx] * sep_power * 0.7
        p_outcome_a[h1_idx] += shift
        p_outcome_a[h2_idx] -= shift
        p_outcome_a = np.clip(p_outcome_a, 1e-6, 1.0)
        p_outcome_a /= p_outcome_a.sum()

        # Counterfactual Outcome B: Evidence supports H2
        p_outcome_b = current_probs.copy()
        shift = p_outcome_b[h1_idx] * sep_power * 0.7
        p_outcome_b[h2_idx] += shift
        p_outcome_b[h1_idx] -= shift
        p_outcome_b = np.clip(p_outcome_b, 1e-6, 1.0)
        p_outcome_b /= p_outcome_b.sum()

        eps = 1e-8
        ent_a = -float(np.sum(p_outcome_a * np.log2(p_outcome_a + eps)))
        ent_b = -float(np.sum(p_outcome_b * np.log2(p_outcome_b + eps)))
        ent_init = -float(np.sum(current_probs * np.log2(current_probs + eps)))

        # Expected uncertainty reduction under counterfactual distribution
        prob_h1 = float(current_probs[h1_idx])
        prob_h2 = float(current_probs[h2_idx])
        norm_sum = prob_h1 + prob_h2 if (prob_h1 + prob_h2) > 0 else 1.0
        w1, w2 = prob_h1 / norm_sum, prob_h2 / norm_sum

        exp_entropy = w1 * ent_a + w2 * ent_b
        expected_entropy_reduction = max(0.0, ent_init - exp_entropy)

        return {
            "evidence_type": evidence_type,
            "hypothesis_separation": sep_power,
            "expected_entropy_reduction": expected_entropy_reduction,
            "p_outcome_if_h1": p_outcome_a,
            "p_outcome_if_h2": p_outcome_b
        }
