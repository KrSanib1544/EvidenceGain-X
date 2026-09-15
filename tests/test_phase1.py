import pytest
from pathlib import Path
from src.data.manifest import DatasetManifest
from src.data.inspector import DatasetInspector

def test_manifest_inspection():
    manifest_path = "data/manifests/dataset_manifest.json"
    assert Path(manifest_path).exists()
    
    manifest = DatasetManifest.load_json(manifest_path)
    assert manifest.crop == "tomato"
    assert len(manifest.classes) == 4
    assert len(manifest.cases) > 0

    inspector = DatasetInspector(manifest)
    summary = inspector.summary()
    assert summary["total_cases"] == len(manifest.cases)
    assert summary["crop"] == "tomato"
    assert "early_blight" in summary["class_distribution"]
    
    # Check duplicate detection
    dupes = inspector.check_duplicates()
    assert isinstance(dupes, list)

    # Check leakage analysis
    leakage = inspector.check_leakage_risks()
    assert leakage["total_inconsistent_plants"] == 0
