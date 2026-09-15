import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import DatasetManifest
from src.data.episodes import EpisodeBuilder
from src.models.classifier import DiseaseClassifier
from src.diagnosis.pipeline import EvidenceGainXPipeline
from src.evidence.policies import InitialOnlyPolicy, RandomPolicy, FixedOrderPolicy
from src.evidence.selector import EvidenceGainXPolicy
from src.verification.reliability import EvidenceReliabilityChecker
from src.verification.contradiction import ContradictionVerifier

def test_full_pipeline_flow():
    classes = ["early_blight", "late_blight", "septoria_leaf_spot", "healthy"]
    cand_types = ["lesion_closeup", "leaf_underside", "stem_view", "symptom_text", "weather_context"]
    
    manifest_path = "data/manifests/dataset_manifest.json"
    assert Path(manifest_path).exists()
    
    manifest = DatasetManifest.load_json(manifest_path)
    sample_case = manifest.cases[0]

    classifier = DiseaseClassifier(num_classes=len(classes), pretrained=False)
    policy = EvidenceGainXPolicy(classes=classes, candidate_types=cand_types)
    pipeline = EvidenceGainXPipeline(
        classifier=classifier,
        policy=policy,
        classes=classes,
        max_budget=3
    )

    # 1. Provisional diagnosis
    init_state = pipeline.diagnose_initial(sample_case.initial_image_path)
    assert "leading_disease" in init_state
    assert "entropy" in init_state
    assert "margin" in init_state

    # 2. Run full episode
    final_state = pipeline.run_episode(sample_case.initial_image_path, sample_case.available_evidence)
    assert "final_diagnosis" in final_state
    assert "total_steps_taken" in final_state
    assert len(final_state["evidence_trail"]) > 0

def test_reliability_checker():
    checker = EvidenceReliabilityChecker()
    manifest = DatasetManifest.load_json("data/manifests/dataset_manifest.json")
    sample_ev = manifest.cases[0].available_evidence["lesion_closeup"]
    
    report = checker.assess_evidence(sample_ev)
    assert report.quality_score >= 0.0
    assert report.action in ["ACCEPT", "DOWNWEIGHT", "REJECT"]
