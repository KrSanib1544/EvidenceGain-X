import random
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path
from src.data.manifest import DatasetManifest, CaseRecord
from src.utils.seed import set_seed

def create_leakage_safe_splits(
    manifest: DatasetManifest,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[CaseRecord], List[CaseRecord], List[CaseRecord]]:
    set_seed(seed)
    
    # Group cases by (disease, plant_id) to stratify while ensuring plant-level isolation
    disease_to_plants: Dict[str, Dict[str, List[CaseRecord]]] = defaultdict(lambda: defaultdict(list))
    for c in manifest.cases:
        disease_to_plants[c.disease][c.plant_id].append(c)

    train_cases: List[CaseRecord] = []
    val_cases: List[CaseRecord] = []
    test_cases: List[CaseRecord] = []

    for disease, plant_dict in disease_to_plants.items():
        plant_ids = list(plant_dict.keys())
        random.shuffle(plant_ids)

        n_total = len(plant_ids)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        
        train_plants = plant_ids[:n_train]
        val_plants = plant_ids[n_train:n_train + n_val]
        test_plants = plant_ids[n_train + n_val:]
        if not test_plants and len(val_plants) > 1:
            test_plants = [val_plants.pop()]

        for pid in train_plants:
            train_cases.extend(plant_dict[pid])
        for pid in val_plants:
            val_cases.extend(plant_dict[pid])
        for pid in test_plants:
            test_cases.extend(plant_dict[pid])

    return train_cases, val_cases, test_cases

def save_split_manifests(
    train_cases: List[CaseRecord],
    val_cases: List[CaseRecord],
    test_cases: List[CaseRecord],
    manifest: DatasetManifest,
    output_dir: str = "data/manifests"
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split_name, cases in [("train", train_cases), ("val", val_cases), ("test", test_cases)]:
        split_m = DatasetManifest(
            version=manifest.version,
            crop=manifest.crop,
            classes=manifest.classes,
            cases=cases,
            created_at=manifest.created_at,
            metadata={"split": split_name, "total_cases": len(cases)}
        )
        split_m.save_json(str(out_path / f"manifest_{split_name}.json"))
