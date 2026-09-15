from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import hashlib

@dataclass
class EvidenceItem:
    evidence_type: str  # e.g., 'lesion_closeup', 'leaf_underside', 'stem_view', 'symptom_text', 'weather_context'
    modality: str       # 'image', 'text', 'tabular'
    data_path_or_content: str # Relative path to image or raw text/metadata string
    reliability_hint: float = 1.0 # Ground truth quality rating (if annotated)
    cost: float = 1.0 # Effort cost to acquire this observation
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CaseRecord:
    case_id: str
    plant_id: str
    crop: str
    disease: str
    initial_image_path: str
    initial_md5: str
    available_evidence: Dict[str, EvidenceItem] = field(default_factory=dict)
    source_dataset: str = "PlantVillage_Tomato"
    license: str = "CC-BY-4.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetManifest:
    version: str
    crop: str
    classes: List[str]
    cases: List[CaseRecord] = field(default_factory=list)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "DatasetManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = []
        for c in data.get("cases", []):
            ev_dict = {}
            for k, ev in c.get("available_evidence", {}).items():
                ev_dict[k] = EvidenceItem(**ev)
            c["available_evidence"] = ev_dict
            cases.append(CaseRecord(**c))
        data["cases"] = cases
        return cls(**data)

def compute_file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
