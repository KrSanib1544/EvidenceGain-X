from typing import List, Dict, Any
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.data.episodes import DiagnosticEpisode
from src.diagnosis.pipeline import EvidenceGainXPipeline
from src.models.classifier import DiseaseClassifier
from src.evidence.selector import EvidenceGainXPolicy
from src.utils.logger import setup_logger

class AblationStudyRunner:
    def __init__(
        self,
        classifier: DiseaseClassifier,
        classes: List[str],
        candidate_types: List[str],
        device: str = "cpu"
    ):
        self.classifier = classifier
        self.classes = classes
        self.candidate_types = candidate_types
        self.device = device
        self.logger = setup_logger("AblationRunner")

    def run_ablations(self, test_episodes: List[DiagnosticEpisode]) -> Dict[str, Any]:
        self.logger.info("Executing systematic ablation studies...")

        configs = {
            "Full_EvidenceGainX": {
                "alpha_sep": 1.0, "beta_unc": 0.8, "gamma_diag": 1.2, "lambda_cost": 0.3, "mu_risk": 0.5
            },
            "No_HypothesisSeparation (alpha=0)": {
                "alpha_sep": 0.0, "beta_unc": 0.8, "gamma_diag": 1.2, "lambda_cost": 0.3, "mu_risk": 0.5
            },
            "No_UncertaintyReduction (beta=0)": {
                "alpha_sep": 1.0, "beta_unc": 0.0, "gamma_diag": 1.2, "lambda_cost": 0.3, "mu_risk": 0.5
            },
            "No_DiagnosticImprovement (gamma=0)": {
                "alpha_sep": 1.0, "beta_unc": 0.8, "gamma_diag": 0.0, "lambda_cost": 0.3, "mu_risk": 0.5
            },
            "No_CostPenalty (lambda=0)": {
                "alpha_sep": 1.0, "beta_unc": 0.8, "gamma_diag": 1.2, "lambda_cost": 0.0, "mu_risk": 0.5
            },
            "No_RiskPenalty (mu=0)": {
                "alpha_sep": 1.0, "beta_unc": 0.8, "gamma_diag": 1.2, "lambda_cost": 0.3, "mu_risk": 0.0
            }
        }

        ablation_results = {}

        for config_name, params in configs.items():
            policy = EvidenceGainXPolicy(
                classes=self.classes,
                candidate_types=self.candidate_types,
                **params
            )
            pipeline = EvidenceGainXPipeline(
                classifier=self.classifier,
                policy=policy,
                classes=self.classes,
                device=self.device
            )

            y_true, y_pred = [], []
            steps = []
            for ep in test_episodes:
                res = pipeline.run_episode(ep.initial_image_path, ep.available_candidates)
                y_true.append(ep.true_disease)
                y_pred.append(res["final_diagnosis"])
                steps.append(res["total_steps_taken"])

            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            avg_steps = float(np.mean(steps))

            ablation_results[config_name] = {
                "accuracy": acc,
                "macro_f1": f1,
                "mean_steps": avg_steps
            }

            self.logger.info(f"[{config_name}] Acc: {acc:.4f} | F1: {f1:.4f} | Steps: {avg_steps:.2f}")

        return ablation_results
