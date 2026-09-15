from typing import Dict, Any, List, Set
from collections import Counter
from pathlib import Path
from src.data.manifest import DatasetManifest, compute_file_md5

class DatasetInspector:
    def __init__(self, manifest: DatasetManifest):
        self.manifest = manifest

    def summary(self) -> Dict[str, Any]:
        total_cases = len(self.manifest.cases)
        disease_counts = Counter(c.disease for c in self.manifest.cases)
        plant_ids = set(c.plant_id for c in self.manifest.cases)
        
        evidence_availability: Dict[str, int] = Counter()
        for c in self.manifest.cases:
            for ev_type in c.available_evidence.keys():
                evidence_availability[ev_type] += 1

        return {
            "version": self.manifest.version,
            "crop": self.manifest.crop,
            "total_cases": total_cases,
            "unique_plants": len(plant_ids),
            "class_distribution": dict(disease_counts),
            "evidence_coverage": dict(evidence_availability),
            "licenses": list(set(c.license for c in self.manifest.cases)),
            "source_datasets": list(set(c.source_dataset for c in self.manifest.cases))
        }

    def check_duplicates(self) -> List[Dict[str, Any]]:
        hash_to_cases: Dict[str, List[str]] = {}
        for c in self.manifest.cases:
            h = c.initial_md5
            if not h and Path(c.initial_image_path).exists():
                h = compute_file_md5(c.initial_image_path)
            hash_to_cases.setdefault(h, []).append(c.case_id)

        duplicates = [
            {"md5": h, "case_ids": ids, "count": len(ids)}
            for h, ids in hash_to_cases.items()
            if len(ids) > 1
        ]
        return duplicates

    def check_leakage_risks(self) -> Dict[str, Any]:
        plant_to_diseases: Dict[str, Set[str]] = {}
        for c in self.manifest.cases:
            plant_to_diseases.setdefault(c.plant_id, set()).add(c.disease)
        
        inconsistent_plants = {
            pid: list(dis) for pid, dis in plant_to_diseases.items() if len(dis) > 1
        }
        return {
            "inconsistent_plant_labels": inconsistent_plants,
            "total_inconsistent_plants": len(inconsistent_plants)
        }
