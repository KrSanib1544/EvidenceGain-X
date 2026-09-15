from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from src.data.manifest import CaseRecord, EvidenceItem

@dataclass
class DiagnosticEpisode:
    case_id: str
    plant_id: str
    true_disease: str
    true_class_idx: int
    initial_image_path: str
    available_candidates: Dict[str, EvidenceItem]
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_terminal: bool = False
    current_step: int = 0
    max_budget: int = 3

    def acquire_evidence(self, evidence_type: str) -> Optional[EvidenceItem]:
        if evidence_type not in self.available_candidates:
            return None
        return self.available_candidates[evidence_type]

class EpisodeBuilder:
    def __init__(self, classes: List[str], max_budget: int = 3):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.max_budget = max_budget

    def build_episode(self, case: CaseRecord) -> DiagnosticEpisode:
        return DiagnosticEpisode(
            case_id=case.case_id,
            plant_id=case.plant_id,
            true_disease=case.disease,
            true_class_idx=self.class_to_idx[case.disease],
            initial_image_path=case.initial_image_path,
            available_candidates=dict(case.available_evidence),
            max_budget=self.max_budget
        )

    def build_episodes_from_cases(self, cases: List[CaseRecord]) -> List[DiagnosticEpisode]:
        return [self.build_episode(c) for c in cases]
