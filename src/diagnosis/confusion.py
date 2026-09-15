from typing import List, Dict, Any, Optional
import numpy as np
import torch

class DiseaseConfusionAnalyzer:
    def __init__(
        self,
        classes: List[str],
        confusion_matrix: Optional[np.ndarray] = None,
        embed_dim: int = 16
    ):
        self.classes = classes
        self.num_classes = len(classes)
        self.embed_dim = embed_dim
        
        if confusion_matrix is not None:
            # Row-normalize confusion matrix
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.confusion_profile = confusion_matrix / row_sums
        else:
            self.confusion_profile = np.eye(self.num_classes)

        # Build pairwise disease-conflict lookup table
        self.conflict_vectors = np.zeros((self.num_classes, self.num_classes, self.embed_dim), dtype=np.float32)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    vec = np.zeros(self.embed_dim, dtype=np.float32)
                    vec[i % self.embed_dim] = 1.0
                    vec[j % self.embed_dim] = -1.0
                    conf_weight = float(self.confusion_profile[i, j] + self.confusion_profile[j, i]) / 2.0
                    self.conflict_vectors[i, j] = vec * (1.0 + conf_weight)

    def get_conflict_representation(self, class_idx_1: int, class_idx_2: int) -> torch.Tensor:
        vec = self.conflict_vectors[class_idx_1, class_idx_2]
        return torch.tensor(vec, dtype=torch.float32)

    def characterize_conflict(self, top_hypotheses: List[Any]) -> Dict[str, Any]:
        if len(top_hypotheses) < 2:
            return {"has_conflict": False, "conflict_repr": torch.zeros(self.embed_dim)}

        h1 = top_hypotheses[0]
        h2 = top_hypotheses[1]
        
        c_repr = self.get_conflict_representation(h1.class_idx, h2.class_idx)
        pair_confusion = float(self.confusion_profile[h1.class_idx, h2.class_idx])

        return {
            "has_conflict": True,
            "disease_pair": (h1.disease, h2.disease),
            "pair_indices": (h1.class_idx, h2.class_idx),
            "pair_confusion_rate": pair_confusion,
            "conflict_repr": c_repr
        }
