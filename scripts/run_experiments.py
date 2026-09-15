import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.manifest import DatasetManifest
from src.data.splitter import create_leakage_safe_splits, save_split_manifests
from src.data.transforms import get_transforms
from src.data.dataset import CropDiseaseDataset
from src.data.episodes import EpisodeBuilder
from src.models.classifier import DiseaseClassifier
from src.models.trainer import ClassifierTrainer
from src.evaluation.benchmark import PolicyBenchmarkRunner
from src.evaluation.ablations import AblationStudyRunner

def run_all_experiments(config_path: str = "configs/config.yaml"):
    logger = setup_logger("EvidenceGainX_ExperimentEngine")
    logger.info("Starting EvidenceGain-X End-to-End Experimental Suite...")

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    manifest_path = "data/manifests/dataset_manifest.json"
    if not Path(manifest_path).exists():
        logger.info("Manifest not found. Generating benchmark dataset first...")
        from scripts.download_data import create_synthetic_benchmark_dataset
        manifest = create_synthetic_benchmark_dataset()
    else:
        manifest = DatasetManifest.load_json(manifest_path)

    classes = cfg["dataset"]["classes"]
    cand_types = cfg["evidence"]["candidate_types"]

    # 1. Leakage-safe plant-grouped splits
    train_cases, val_cases, test_cases = create_leakage_safe_splits(
        manifest=manifest,
        train_ratio=cfg["dataset"]["split_ratios"]["train"],
        val_ratio=cfg["dataset"]["split_ratios"]["val"],
        test_ratio=cfg["dataset"]["split_ratios"]["test"],
        seed=cfg.get("seed", 42)
    )
    save_split_manifests(train_cases, val_cases, test_cases, manifest)
    logger.info(f"Dataset split complete: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)} cases.")

    # 2. Create PyTorch datasets and loaders
    train_tf, eval_tf = get_transforms(img_size=cfg["dataset"]["img_size"])
    train_ds = CropDiseaseDataset(train_cases, classes, transform=train_tf)
    val_ds = CropDiseaseDataset(val_cases, classes, transform=eval_tf)
    test_ds = CropDiseaseDataset(test_cases, classes, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=False)

    # 3. Baseline Classifier Training & Calibration
    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    logger.info(f"Execution device: {device}")

    classifier = DiseaseClassifier(
        num_classes=len(classes),
        backbone_name=cfg["classifier"]["backbone"],
        pretrained=False # Set to False for local offline reproducible testing
    )
    trainer = ClassifierTrainer(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg["classifier"]["lr"],
        weight_decay=cfg["classifier"]["weight_decay"]
    )
    train_results = trainer.fit(epochs=cfg["classifier"]["epochs"])

    # 4. Build Test Episodes
    ep_builder = EpisodeBuilder(classes=classes, max_budget=cfg["evidence"]["max_budget"])
    test_episodes = ep_builder.build_episodes_from_cases(test_cases)

    # 5. Baseline Benchmarking (B0 to B6)
    bench_runner = PolicyBenchmarkRunner(
        classifier=classifier,
        classes=classes,
        candidate_types=cand_types,
        device=device
    )
    bench_report = bench_runner.run_benchmark(test_episodes)
    bench_report_path = "results/benchmark_results.json"
    bench_report.save_json(bench_report_path)
    logger.info(f"Benchmark results saved to {bench_report_path}")

    # 6. Ablation Studies
    ablation_runner = AblationStudyRunner(
        classifier=classifier,
        classes=classes,
        candidate_types=cand_types,
        device=device
    )
    ablation_results = ablation_runner.run_ablations(test_episodes)
    ablation_path = Path("results/ablation_results.json")
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info(f"Ablation results saved to {ablation_path}")

    logger.info("=== EVIDENCEGAIN-X EXPERIMENTAL SUMMARY ===")
    for p_name, res in bench_report.results.items():
        logger.info(
            f"[{p_name:20s}] Acc: {res.accuracy:.4f} | F1: {res.macro_f1:.4f} | "
            f"Steps: {res.mean_steps:.2f} | Delta_Entropy: {res.mean_entropy_reduction:.4f}"
        )

if __name__ == "__main__":
    run_all_experiments()
