from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.data.episodes import DiagnosticEpisode
from src.diagnosis.pipeline import EvidenceGainXPipeline
from src.models.classifier import DiseaseClassifier
from src.evidence.policies import (
    BaseAcquisitionPolicy,
    InitialOnlyPolicy,
    RandomPolicy,
    FixedOrderPolicy,
    ConfidenceDrivenPolicy,
    UncertaintyOnlyPolicy
)
from src.evidence.selector import EvidenceGainXPolicy
from src.utils.logger import setup_logger

@dataclass
class PolicyResult:
    policy_name: str
    accuracy: float
    macro_f1: float
    mean_entropy_initial: float
    mean_entropy_final: float
    mean_entropy_reduction: float
    mean_steps: float
    stopping_rate: float
    rejection_count: int
    contradiction_count: int

@dataclass
class BenchmarkReport:
    total_episodes: int
    classes: List[str]
    results: Dict[str, PolicyResult]
    budget_curve: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def save_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Convert dataclass to dict
        data = {
            "total_episodes": self.total_episodes,
            "classes": self.classes,
            "results": {k: asdict(v) for k, v in self.results.items()},
            "budget_curve": self.budget_curve
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

class PolicyBenchmarkRunner:
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
        self.logger = setup_logger("BenchmarkRunner")

        # Instantiate all standard baselines
        self.policies: Dict[str, BaseAcquisitionPolicy] = {
            "B0_InitialOnly": InitialOnlyPolicy(),
            "B1_Random": RandomPolicy(seed=42),
            "B2_FixedOrder": FixedOrderPolicy(default_order=candidate_types),
            "B3_ConfidenceDriven": ConfidenceDrivenPolicy(conf_threshold=0.75),
            "B4_UncertaintyOnly": UncertaintyOnlyPolicy(entropy_threshold=0.50),
            "B6_EvidenceGainX": EvidenceGainXPolicy(classes=classes, candidate_types=candidate_types)
        }

    def run_benchmark(self, test_episodes: List[DiagnosticEpisode]) -> BenchmarkReport:
        self.logger.info(f"Running comprehensive policy benchmark on {len(test_episodes)} test episodes...")
        results: Dict[str, PolicyResult] = {}
        budget_curves: Dict[str, Dict[int, float]] = {}

        for p_name, policy in self.policies.items():
            pipeline = EvidenceGainXPipeline(
                classifier=self.classifier,
                policy=policy,
                classes=self.classes,
                device=self.device,
                max_budget=3
            )

            y_true = []
            y_pred = []
            init_entropies = []
            final_entropies = []
            steps_list = []
            rejections = 0
            contradictions = 0
            stopped_early = 0

            for ep in test_episodes:
                res = pipeline.run_episode(ep.initial_image_path, ep.available_candidates)
                
                y_true.append(ep.true_disease)
                y_pred.append(res["final_diagnosis"])
                
                init_entropies.append(float(-np.sum(res["initial_probs"] * np.log2(res["initial_probs"] + 1e-8))))
                final_entropies.append(res["entropy"])
                steps_list.append(res["total_steps_taken"])
                if res["is_sufficient"]:
                    stopped_early += 1

                for trail_item in res["evidence_trail"]:
                    if trail_item.get("action") == "REJECTED":
                        rejections += 1
                    if trail_item.get("contradiction"):
                        contradictions += 1

            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            m_init_e = float(np.mean(init_entropies))
            m_fin_e = float(np.mean(final_entropies))
            m_steps = float(np.mean(steps_list))

            results[p_name] = PolicyResult(
                policy_name=p_name,
                accuracy=acc,
                macro_f1=f1,
                mean_entropy_initial=m_init_e,
                mean_entropy_final=m_fin_e,
                mean_entropy_reduction=m_init_e - m_fin_e,
                mean_steps=m_steps,
                stopping_rate=float(stopped_early / len(test_episodes)),
                rejection_count=rejections,
                contradiction_count=contradictions
            )

            self.logger.info(
                f"[{p_name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | "
                f"Avg Steps: {m_steps:.2f} | Delta_Entropy: {m_init_e - m_fin_e:.4f}"
            )

        return BenchmarkReport(
            total_episodes=len(test_episodes),
            classes=self.classes,
            results=results,
            budget_curve=budget_curves
        )
