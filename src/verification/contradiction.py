from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import torch

@dataclass
class ContradictionReport:
    is_contradiction: bool
    contradiction_score: float # 0.0 to 1.0 (higher = more contradictory)
    expected_shift_direction: str
    observed_shift_direction: str
    penalty_factor: float
    explanation: str

class ContradictionVerifier:
    def __init__(self, contradiction_threshold: float = 0.40, penalty_scale: float = 0.50):
        self.contradiction_threshold = contradiction_threshold
        self.penalty_scale = penalty_scale

    def verify(
        self,
        prior_probs: np.ndarray,
        expected_probs: np.ndarray,
        observed_probs: np.ndarray,
        h1_idx: int,
        h2_idx: int,
        classes: List[str]
    ) -> ContradictionReport:
        # Compute direction of expected shift on leading hypothesis vs rival
        expected_diff = (expected_probs[h1_idx] - prior_probs[h1_idx]) - (expected_probs[h2_idx] - prior_probs[h2_idx])
        observed_diff = (observed_probs[h1_idx] - prior_probs[h1_idx]) - (observed_probs[h2_idx] - prior_probs[h2_idx])

        # Cosine / directional divergence between expected update and observed update
        delta_exp = expected_probs - prior_probs
        delta_obs = observed_probs - prior_probs

        norm_exp = np.linalg.norm(delta_exp)
        norm_obs = np.linalg.norm(delta_obs)

        if norm_exp < 1e-6 or norm_obs < 1e-6:
            cosine_sim = 1.0
        else:
            cosine_sim = float(np.dot(delta_exp, delta_obs) / (norm_exp * norm_obs))

        # Contradiction score ranges from 0 (perfect alignment) to 1 (complete opposition)
        contradiction_score = float(np.clip((1.0 - cosine_sim) / 2.0, 0.0, 1.0))
        is_contradiction = contradiction_score >= self.contradiction_threshold

        h1_name = classes[h1_idx]
        h2_name = classes[h2_idx]

        exp_dir = f"Support {h1_name}" if expected_diff > 0 else f"Support {h2_name}"
        obs_dir = f"Support {h1_name}" if observed_diff > 0 else f"Support {h2_name}"

        if is_contradiction:
            penalty_factor = float(1.0 - self.penalty_scale * contradiction_score)
            explanation = (
                f"Contradiction detected: Acquired evidence shifts belief toward '{obs_dir}' "
                f"while planning expected '{exp_dir}' (divergence: {contradiction_score:.2f}). "
                f"Confidence is penalized to prevent artificial certainty."
            )
        else:
            penalty_factor = 1.0
            explanation = f"Observation consistent with expected diagnostic shift (divergence: {contradiction_score:.2f})."

        return ContradictionReport(
            is_contradiction=is_contradiction,
            contradiction_score=contradiction_score,
            expected_shift_direction=exp_dir,
            observed_shift_direction=obs_dir,
            penalty_factor=penalty_factor,
            explanation=explanation
        )
