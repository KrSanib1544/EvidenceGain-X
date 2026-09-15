from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import random
import numpy as np

class BaseAcquisitionPolicy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_next(
        self,
        diagnostic_state: Dict[str, Any],
        available_candidates: List[str]
    ) -> Optional[str]:
        pass

class InitialOnlyPolicy(BaseAcquisitionPolicy):
    def __init__(self):
        super().__init__("B0_InitialOnly")

    def select_next(
        self,
        diagnostic_state: Dict[str, Any],
        available_candidates: List[str]
    ) -> Optional[str]:
        return None # Never acquires additional evidence

class RandomPolicy(BaseAcquisitionPolicy):
    def __init__(self, seed: Optional[int] = 42):
        super().__init__("B1_Random")
        self.rng = random.Random(seed)

    def select_next(
        self,
        diagnostic_state: Dict[str, Any],
        available_candidates: List[str]
    ) -> Optional[str]:
        if not available_candidates:
            return None
        return self.rng.choice(available_candidates)

class FixedOrderPolicy(BaseAcquisitionPolicy):
    def __init__(self, default_order: Optional[List[str]] = None):
        super().__init__("B2_FixedOrder")
        self.order = default_order or [
            "lesion_closeup",
            "leaf_underside",
            "stem_view",
            "symptom_text",
            "weather_context"
        ]

    def select_next(
        self,
        diagnostic_state: Dict[str, Any],
        available_candidates: List[str]
    ) -> Optional[str]:
        for cand in self.order:
            if cand in available_candidates:
                return cand
        return None

class ConfidenceDrivenPolicy(BaseAcquisitionPolicy):
    def __init__(self, conf_threshold: float = 0.75):
        super().__init__("B3_ConfidenceDriven")
        self.conf_threshold = conf_threshold

    def select_next(
        self,
        diagnostic_state: Dict[str, Any],
        available_candidates: List[str]
    ) -> Optional[str]:
        if not available_candidates:
            return None
        leading_prob = diagnostic_state.get("leading_prob", 1.0)
        if leading_prob >= self.conf_threshold:
            return None # Sufficient confidence, stop
        # Otherwise select first available candidate
        return available_candidates[0]

class UncertaintyOnlyPolicy(BaseAcquisitionPolicy):
    def __init__(self, entropy_threshold: float = 0.5):
        super().__init__("B4_UncertaintyOnly")
        self.entropy_threshold = entropy_threshold
        # Generic expected uncertainty reduction heuristic per candidate type
        self.cand_entropy_reduction_weights = {
            "lesion_closeup": 0.45,
            "leaf_underside": 0.35,
            "stem_view": 0.25,
            "symptom_text": 0.40,
            "weather_context": 0.20
        }

    def select_next(
        self,
        diagnostic_state: Dict[str, Any],
        available_candidates: List[str]
    ) -> Optional[str]:
        if not available_candidates:
            return None
        current_entropy = diagnostic_state.get("entropy", 0.0)
        if current_entropy <= self.entropy_threshold:
            return None # Low uncertainty, stop
        
        # Select candidate with highest generic entropy reduction
        best_cand = max(
            available_candidates,
            key=lambda c: self.cand_entropy_reduction_weights.get(c, 0.1)
        )
        return best_cand
