import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Tuple, Optional

class DiseaseClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "resnet18",
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.embed_dim = in_features
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.embed_dim = in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.classifier_head = nn.Linear(self.embed_dim, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x) # (B, embed_dim)
        logits = self.classifier_head(features) # (B, num_classes)
        return logits, features

    def predict_provisional(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, features = self.forward(x)
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Calculate entropy H(p) = -sum(p * log(p + eps))
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        
        # Sort probabilities for margin computation
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        return {
            "logits": logits,
            "scaled_logits": scaled_logits,
            "probs": probs,
            "features": features,
            "entropy": entropy,
            "margin": margin,
            "top1_pred": sorted_indices[:, 0],
            "top1_prob": sorted_probs[:, 0],
            "sorted_indices": sorted_indices,
            "sorted_probs": sorted_probs
        }
