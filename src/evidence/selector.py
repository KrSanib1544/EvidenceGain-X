from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from src.evidence.policies import BaseAcquisitionPolicy
from src.planning.counterfactual import CounterfactualEvidencePlanner

class UtilityScoringNetwork(nn.Module):
    def __init__(self, state_dim: int, num_candidates: int, hidden_dim: int = 64):
        super().__init__()
        self.cand_embed = nn.Embedding(num_candidates, 16)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state_tensor: torch.Tensor, cand_indices: torch.Tensor) -> torch.Tensor:
        # state_tensor: (B, state_dim), cand_indices: (B,)
        c_emb = self.cand_embed(cand_indices) # (B, 16)
        x = torch.cat([state_tensor, c_emb], dim=-1)
        return self.net(x).squeeze(-1) # (B,)

class EvidenceGainSelector:
    def __init__(
        self,
        candidate_types: List[str],
        state_dim: int = 24,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        self.candidate_types = candidate_types
        self.cand_to_idx = {c: i for i, c in enumerate(candidate_types)}
        self.device = device
        self.model = UtilityScoringNetwork(state_dim, len(candidate_types), hidden_dim).to(device)

    def predict_utility(
        self,
        state_repr: np.ndarray,
        candidate: str
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            s_t = torch.tensor(state_repr, dtype=torch.float32, device=self.device).unsqueeze(0)
            c_idx = torch.tensor([self.cand_to_idx[candidate]], dtype=torch.long, device=self.device)
            score = self.model(s_t, c_idx)
            return float(score.item())

class EvidenceGainXPolicy(BaseAcquisitionPolicy):
    def __init__(
        self,
        classes: List[str],
        candidate_types: List[str],
        selector: Optional[EvidenceGainSelector] = None,
        alpha_sep: float = 1.0,
        beta_unc: float = 0.8,
        gamma_diag: float = 1.2,
        lambda_cost: float = 0.3,
        mu_risk: float = 0.5,
        stopping_entropy: float = 0.35,
        stopping_margin: float = 0.70
    ):
        super().__init__("B6_EvidenceGainX")
        self.classes = classes
        self.candidate_types = candidate_types
        self.selector = selector
        self.planner = CounterfactualEvidencePlanner(classes, candidate_types)
        
        self.alpha_sep = alpha_sep
        self.beta_unc = beta_unc
        self.gamma_diag = gamma_diag
        self.lambda_cost = lambda_cost
        self.mu_risk = mu_risk
        
        self.stopping_entropy = stopping_entropy
        self.stopping_margin = stopping_margin

        # Default acquisition costs and risk priors
        self.costs = {
            "symptom_text": 0.4,
            "weather_context": 0.4,
            "lesion_closeup": 1.0,
            "leaf_underside": 1.2,
            "stem_view": 1.5
        }
        self.risks = {
            "symptom_text": 0.1,
            "weather_context": 0.15,
            "lesion_closeup": 0.2,
            "leaf_underside": 0.3,
            "stem_view": 0.35
        }

    def compute_evidence_utility(
        self,
        current_probs: np.ndarray,
        h1_idx: int,
        h2_idx: int,
        candidate: str
    ) -> Dict[str, float]:
        cf_plan = self.planner.simulate_counterfactual_outcomes(current_probs, h1_idx, h2_idx, candidate)
        
        sep = cf_plan["hypothesis_separation"]
        unc_red = cf_plan["expected_entropy_reduction"]
        
        # Expected diagnostic improvement (expected increase in top probability margin)
        exp_diag_imp = sep * 0.5
        cost = self.costs.get(candidate, 1.0)
        risk = self.risks.get(candidate, 0.2)

        total_utility = (
            self.alpha_sep * sep +
            self.beta_unc * unc_red +
            self.gamma_diag * exp_diag_imp -
            self.lambda_cost * cost -
            self.mu_risk * risk
        )

        return {
            "candidate": candidate,
            "total_utility": float(total_utility),
            "hypothesis_separation": float(sep),
            "uncertainty_reduction": float(unc_red),
            "diagnostic_improvement": float(exp_diag_imp),
            "cost": float(cost),
            "risk": float(risk)
        }

    def select_next(
        self,
        diagnostic_state: Dict[str, Any],
        available_candidates: List[str]
    ) -> Optional[str]:
        if not available_candidates:
            return None

        # Check stopping criteria
        entropy = diagnostic_state.get("entropy", 1.0)
        margin = diagnostic_state.get("margin", 0.0)
        if entropy <= self.stopping_entropy and margin >= self.stopping_margin:
            return None # Sufficient evidence, stop

        # Competing hypotheses indices
        top_hypos = diagnostic_state.get("top_hypotheses", [])
        if len(top_hypos) >= 2:
            h1_idx = top_hypos[0].class_idx
            h2_idx = top_hypos[1].class_idx
        elif len(top_hypos) == 1:
            h1_idx = top_hypos[0].class_idx
            h2_idx = (h1_idx + 1) % len(self.classes)
        else:
            h1_idx, h2_idx = 0, 1

        probs_dict = diagnostic_state.get("full_distribution", {})
        probs_np = np.array([probs_dict.get(c, 0.25) for c in self.classes], dtype=np.float32)

        best_cand = None
        best_u = -1e9

        for cand in available_candidates:
            u_dict = self.compute_evidence_utility(probs_np, h1_idx, h2_idx, cand)
            if u_dict["total_utility"] > best_u:
                best_u = u_dict["total_utility"]
                best_cand = cand

        return best_cand
