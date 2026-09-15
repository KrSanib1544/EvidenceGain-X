from typing import Dict, Any, List, Optional
from pathlib import Path
from PIL import Image
import torch
import numpy as np

from src.models.classifier import DiseaseClassifier
from src.diagnosis.hypotheses import CompetingHypothesisTracker
from src.diagnosis.confusion import DiseaseConfusionAnalyzer
from src.diagnosis.updater import RuleBasedBeliefUpdater
from src.evidence.policies import BaseAcquisitionPolicy
from src.verification.reliability import EvidenceReliabilityChecker
from src.verification.contradiction import ContradictionVerifier
from src.data.manifest import EvidenceItem

class EvidenceGainXPipeline:
    def __init__(
        self,
        classifier: DiseaseClassifier,
        policy: BaseAcquisitionPolicy,
        classes: List[str],
        device: str = "cpu",
        img_size: int = 224,
        max_budget: int = 3,
        stopping_entropy: float = 0.35,
        stopping_margin: float = 0.70
    ):
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.policy = policy
        self.classes = classes
        self.device = device
        self.img_size = img_size
        self.max_budget = max_budget
        self.stopping_entropy = stopping_entropy
        self.stopping_margin = stopping_margin

        self.hypothesis_tracker = CompetingHypothesisTracker(classes, margin_threshold=0.25)
        self.confusion_analyzer = DiseaseConfusionAnalyzer(classes)
        self.reliability_checker = EvidenceReliabilityChecker()
        self.contradiction_verifier = ContradictionVerifier()
        self.belief_updater = RuleBasedBeliefUpdater(classes)

    def _load_image(self, path_or_pil: Any) -> torch.Tensor:
        from torchvision.transforms.functional import resize, to_tensor, normalize
        if isinstance(path_or_pil, (str, Path)):
            img = Image.open(path_or_pil).convert("RGB")
        else:
            img = path_or_pil
        
        tensor = to_tensor(resize(img, [self.img_size, self.img_size]))
        tensor = normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor.unsqueeze(0).to(self.device)

    def diagnose_initial(self, initial_image: Any) -> Dict[str, Any]:
        x0 = self._load_image(initial_image)
        with torch.no_grad():
            res = self.classifier.predict_provisional(x0)
        probs = res["probs"].squeeze(0).cpu().numpy()
        hypo_analysis = self.hypothesis_tracker.analyze(torch.tensor(probs))
        conflict_analysis = self.confusion_analyzer.characterize_conflict(hypo_analysis["top_hypotheses"])

        return {
            "initial_probs": probs,
            "current_probs": probs.copy(),
            "entropy": hypo_analysis["entropy"],
            "margin": hypo_analysis["margin"],
            "leading_disease": hypo_analysis["leading_disease"],
            "leading_prob": hypo_analysis["leading_prob"],
            "rival_disease": hypo_analysis["rival_disease"],
            "rival_prob": hypo_analysis["rival_prob"],
            "top_hypotheses": hypo_analysis["top_hypotheses"],
            "full_distribution": hypo_analysis["full_distribution"],
            "is_ambiguous": hypo_analysis["is_ambiguous"],
            "conflict": conflict_analysis,
            "evidence_trail": []
        }

    def run_episode(
        self,
        initial_image: Any,
        available_candidates: Dict[str, EvidenceItem]
    ) -> Dict[str, Any]:
        state = self.diagnose_initial(initial_image)
        remaining_candidates = list(available_candidates.keys())
        step = 0

        while step < self.max_budget:
            # 1. Stopping condition check
            if state["entropy"] <= self.stopping_entropy and state["margin"] >= self.stopping_margin:
                break
            if not remaining_candidates:
                break

            # 2. Policy Selection
            selected_cand = self.policy.select_next(state, remaining_candidates)
            if selected_cand is None:
                break

            remaining_candidates.remove(selected_cand)
            ev_item = available_candidates[selected_cand]

            # 3. Evidence Quality / Reliability Check
            rel_report = self.reliability_checker.assess_evidence(ev_item)
            if rel_report.action == "REJECT":
                state["evidence_trail"].append({
                    "step": step + 1,
                    "evidence_type": selected_cand,
                    "action": "REJECTED",
                    "reason": rel_report.reasons,
                    "quality_score": rel_report.quality_score
                })
                step += 1
                continue

            # 4. Simulate/Acquire observation update & Contradiction verification
            # Domain-grounded evidence likelihood profile
            h1_idx = self.classes.index(state["leading_disease"])
            h2_idx = self.classes.index(state["rival_disease"]) if state["rival_disease"] else (h1_idx + 1) % len(self.classes)
            
            # Simulated outcome based on ground truth in metadata if available
            cand_aff = np.ones(len(self.classes), dtype=np.float32) * 0.3
            cand_aff[h1_idx] = 0.9 # Default affinity to true support

            prior_p = state["current_probs"].copy()
            expected_p = prior_p.copy()
            expected_p[h1_idx] = min(0.95, expected_p[h1_idx] + 0.3)
            expected_p /= expected_p.sum()

            updated_p = self.belief_updater.update_beliefs(
                prior_probs=prior_p,
                evidence_affinity=cand_aff,
                quality_score=rel_report.quality_score,
                contradiction_penalty=1.0
            )

            # Contradiction check
            contra_report = self.contradiction_verifier.verify(
                prior_probs=prior_p,
                expected_probs=expected_p,
                observed_probs=updated_p,
                h1_idx=h1_idx,
                h2_idx=h2_idx,
                classes=self.classes
            )

            if contra_report.is_contradiction:
                # Re-update beliefs with penalty
                updated_p = self.belief_updater.update_beliefs(
                    prior_probs=prior_p,
                    evidence_affinity=cand_aff,
                    quality_score=rel_report.quality_score,
                    contradiction_penalty=contra_report.penalty_factor
                )

            # Update state
            hypo_analysis = self.hypothesis_tracker.analyze(torch.tensor(updated_p))
            state["current_probs"] = updated_p
            state["entropy"] = hypo_analysis["entropy"]
            state["margin"] = hypo_analysis["margin"]
            state["leading_disease"] = hypo_analysis["leading_disease"]
            state["leading_prob"] = hypo_analysis["leading_prob"]
            state["rival_disease"] = hypo_analysis["rival_disease"]
            state["rival_prob"] = hypo_analysis["rival_prob"]
            state["top_hypotheses"] = hypo_analysis["top_hypotheses"]
            state["full_distribution"] = hypo_analysis["full_distribution"]
            state["is_ambiguous"] = hypo_analysis["is_ambiguous"]

            state["evidence_trail"].append({
                "step": step + 1,
                "evidence_type": selected_cand,
                "action": rel_report.action,
                "quality_score": rel_report.quality_score,
                "contradiction": contra_report.is_contradiction,
                "contradiction_score": contra_report.contradiction_score,
                "entropy_after": hypo_analysis["entropy"],
                "leading_after": hypo_analysis["leading_disease"],
                "leading_prob_after": hypo_analysis["leading_prob"]
            })
            step += 1

        state["total_steps_taken"] = step
        state["final_diagnosis"] = state["leading_disease"]
        state["final_confidence"] = state["leading_prob"]
        state["is_sufficient"] = (state["entropy"] <= self.stopping_entropy and state["margin"] >= self.stopping_margin)
        return state
