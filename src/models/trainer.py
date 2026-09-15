import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

from src.models.classifier import DiseaseClassifier
from src.models.calibration import TemperatureScaler, calculate_ece
from src.utils.logger import setup_logger

class ClassifierTrainer:
    def __init__(
        self,
        model: DiseaseClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        logger_name: str = "ClassifierTrainer"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = setup_logger(logger_name)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_val_f1 = 0.0
        self.best_state_dict = None

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        if loader is None:
            loader = self.val_loader
        self.model.eval()

        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_preds: List[int] = []

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits, _ = self.model(images)
            preds = torch.argmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_preds.extend(preds.cpu().tolist())

        cat_logits = torch.cat(all_logits, dim=0)
        cat_labels = torch.cat(all_labels, dim=0)
        y_true = cat_labels.numpy()
        y_pred = np.array(all_preds)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        probs = torch.softmax(cat_logits, dim=-1)
        ece = calculate_ece(probs, cat_labels)

        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "ece": float(ece),
            "confusion_matrix": cm.tolist(),
            "logits": cat_logits,
            "labels": cat_labels
        }

    def fit(self, epochs: int = 10, checkpoint_path: str = "checkpoints/classifier_best.pt") -> Dict[str, Any]:
        self.logger.info(f"Starting classifier training for {epochs} epochs on {self.device}...")
        history = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["macro_f1"],
                "val_ece": val_metrics["ece"]
            })

            self.logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val ECE: {val_metrics['ece']:.4f}"
            )

            if val_metrics["macro_f1"] >= self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.best_state_dict = self.model.state_dict()
                
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_metrics": val_metrics,
                    "temperature": self.model.temperature.item()
                }, checkpoint_path)

        # Calibrate with temperature scaling on validation set
        if self.best_state_dict:
            self.model.load_state_dict(self.best_state_dict)
        val_eval = self.evaluate(self.val_loader)
        scaler = TemperatureScaler()
        best_t = scaler.fit(val_eval["logits"], val_eval["labels"])
        self.model.temperature.data.fill_(best_t)
        self.logger.info(f"Calibrated model with Temperature = {best_t:.4f}")

        return {"history": history, "best_val_f1": self.best_val_f1, "calibrated_temp": best_t}
