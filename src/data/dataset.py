from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.data.manifest import DatasetManifest, CaseRecord

class CropDiseaseDataset(Dataset):
    def __init__(
        self,
        cases: List[CaseRecord],
        classes: List[str],
        transform: Optional[Callable] = None
    ):
        self.cases = cases
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]
        img_path = Path(case.initial_image_path)
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
        else:
            from torchvision.transforms.functional import to_tensor
            img_tensor = to_tensor(img)

        label_idx = self.class_to_idx[case.disease]

        return {
            "image": img_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "case_id": case.case_id,
            "plant_id": case.plant_id,
            "disease": case.disease
        }
