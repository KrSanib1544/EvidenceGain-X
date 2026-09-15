from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

@dataclass
class ReliabilityReport:
    is_acceptable: bool
    quality_score: float # 0.0 to 1.0
    action: str          # 'ACCEPT', 'DOWNWEIGHT', 'REJECT'
    blur_score: float
    brightness_score: float
    reasons: list

class EvidenceReliabilityChecker:
    def __init__(
        self,
        blur_threshold: float = 80.0,
        min_brightness: float = 30.0,
        max_brightness: float = 230.0
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def assess_image_quality(self, image_path: str) -> ReliabilityReport:
        p = Path(image_path)
        if not p.exists():
            return ReliabilityReport(
                is_acceptable=False,
                quality_score=0.0,
                action="REJECT",
                blur_score=0.0,
                brightness_score=0.0,
                reasons=["Image file not found"]
            )

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            return ReliabilityReport(
                is_acceptable=False,
                quality_score=0.0,
                action="REJECT",
                blur_score=0.0,
                brightness_score=0.0,
                reasons=["Corrupt or unreadable image format"]
            )

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Variance of Laplacian for blur detection
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # 2. Mean brightness check
        mean_brightness = float(np.mean(gray))

        reasons = []
        is_blurry = blur_score < self.blur_threshold
        is_too_dark = mean_brightness < self.min_brightness
        is_too_bright = mean_brightness > self.max_brightness

        if is_blurry:
            reasons.append(f"Image is blurry (Laplacian variance: {blur_score:.1f} < {self.blur_threshold})")
        if is_too_dark:
            reasons.append(f"Image is underexposed/too dark (brightness: {mean_brightness:.1f})")
        if is_too_bright:
            reasons.append(f"Image is overexposed/too bright (brightness: {mean_brightness:.1f})")

        # Compute normalized quality score
        blur_factor = min(1.0, blur_score / max(1.0, self.blur_threshold * 1.5))
        bright_factor = 1.0
        if mean_brightness < self.min_brightness:
            bright_factor = max(0.1, mean_brightness / self.min_brightness)
        elif mean_brightness > self.max_brightness:
            bright_factor = max(0.1, (255.0 - mean_brightness) / (255.0 - self.max_brightness))

        quality_score = float(np.clip(0.6 * blur_factor + 0.4 * bright_factor, 0.0, 1.0))

        if quality_score >= 0.70 and not is_blurry:
            action = "ACCEPT"
            is_acceptable = True
        elif quality_score >= 0.35:
            action = "DOWNWEIGHT"
            is_acceptable = True
        else:
            action = "REJECT"
            is_acceptable = False

        return ReliabilityReport(
            is_acceptable=is_acceptable,
            quality_score=quality_score,
            action=action,
            blur_score=blur_score,
            brightness_score=mean_brightness,
            reasons=reasons
        )

    def assess_evidence(self, evidence_item: Any) -> ReliabilityReport:
        if getattr(evidence_item, "modality", "image") == "image":
            return self.assess_image_quality(str(evidence_item.data_path_or_content))
        else:
            # Text or tabular evidence: check non-empty string
            content = str(evidence_item.data_path_or_content).strip()
            if len(content) < 3:
                return ReliabilityReport(
                    is_acceptable=False,
                    quality_score=0.1,
                    action="REJECT",
                    blur_score=100.0,
                    brightness_score=128.0,
                    reasons=["Text evidence is empty or too short"]
                )
            return ReliabilityReport(
                is_acceptable=True,
                quality_score=1.0,
                action="ACCEPT",
                blur_score=100.0,
                brightness_score=128.0,
                reasons=[]
            )
