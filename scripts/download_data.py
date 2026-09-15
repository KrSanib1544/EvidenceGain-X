import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from src.data.manifest import DatasetManifest, CaseRecord, EvidenceItem, compute_file_md5
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.utils.logger import setup_logger

def create_synthetic_benchmark_dataset(output_dir: str = "data/raw", num_cases_per_class: int = 40) -> DatasetManifest:
    set_seed(42)
    logger = setup_logger("DataAcquisition")
    logger.info("Initializing benchmark multi-view crop dataset generator...")

    raw_path = Path(output_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    classes = ["early_blight", "late_blight", "septoria_leaf_spot", "healthy"]
    cases: list[CaseRecord] = []

    disease_symptoms = {
        "early_blight": {
            "color": (100, 70, 40),
            "pattern": "concentric_rings",
            "text": "Concentric dark brown rings with chlorotic halo on lower leaves",
            "weather": "Warm 24-29C with heavy dew or frequent rain"
        },
        "late_blight": {
            "color": (50, 60, 40),
            "pattern": "water_soaked",
            "text": "Large irregular water-soaked pale to dark brown lesions with pale margin",
            "weather": "Cool 15-20C with high relative humidity (>90%)"
        },
        "septoria_leaf_spot": {
            "color": (120, 110, 90),
            "pattern": "small_circular_spots",
            "text": "Multiple small circular spots (1-3mm) with dark borders and grey center",
            "weather": "Moderate 20-25C with prolonged leaf wetness"
        },
        "healthy": {
            "color": (34, 139, 34),
            "pattern": "uniform_green",
            "text": "Uniform green leaf tissue with no necrotic spots or chlorosis",
            "weather": "Optimal growing conditions"
        }
    }

    for d_idx, disease in enumerate(classes):
        d_info = disease_symptoms[disease]
        for i in range(num_cases_per_class):
            case_id = f"tomato_{disease}_{i+1:04d}"
            plant_id = f"plant_{disease}_{(i // 3) + 1:03d}" # 3 observations per distinct plant
            
            case_dir = raw_path / case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            # 1. Initial Leaf Image (224x224 RGB)
            init_img = Image.new("RGB", (224, 224), color=(34, 139, 34))
            draw = ImageDraw.Draw(init_img)
            # Add leaf vein structures
            draw.line([(112, 10), (112, 214)], fill=(46, 160, 46), width=3)
            draw.line([(112, 60), (40, 90)], fill=(46, 160, 46), width=2)
            draw.line([(112, 120), (184, 150)], fill=(46, 160, 46), width=2)

            if disease != "healthy":
                for _ in range(5):
                    cx, cy = np.random.randint(40, 180), np.random.randint(40, 180)
                    r = np.random.randint(10, 25)
                    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=d_info["color"])
            
            init_path = str(case_dir / "initial_view.jpg")
            init_img.save(init_path, quality=95)
            init_md5 = compute_file_md5(init_path)

            # 2. Candidate E1: Lesion Close-up
            e1_img = init_img.crop((60, 60, 164, 164)).resize((224, 224), Image.Resampling.BILINEAR)
            e1_path = str(case_dir / "lesion_closeup.jpg")
            e1_img.save(e1_path, quality=95)

            # 3. Candidate E2: Leaf Underside
            e2_img = init_img.filter(ImageFilter.SMOOTH)
            e2_path = str(case_dir / "leaf_underside.jpg")
            e2_img.save(e2_path, quality=95)

            # 4. Candidate E3: Stem View
            e3_img = Image.new("RGB", (224, 224), color=(60, 140, 60))
            e3_draw = ImageDraw.Draw(e3_img)
            e3_draw.rectangle([90, 0, 134, 224], fill=(70, 150, 70))
            if disease in ["early_blight", "late_blight"]:
                e3_draw.rectangle([95, 80, 130, 140], fill=d_info["color"])
            e3_path = str(case_dir / "stem_view.jpg")
            e3_img.save(e3_path, quality=95)

            ev_items = {
                "lesion_closeup": EvidenceItem(
                    evidence_type="lesion_closeup",
                    modality="image",
                    data_path_or_content=e1_path,
                    reliability_hint=0.95,
                    cost=1.2,
                    metadata={"view": "macro_crop", "resolution": [224, 224]}
                ),
                "leaf_underside": EvidenceItem(
                    evidence_type="leaf_underside",
                    modality="image",
                    data_path_or_content=e2_path,
                    reliability_hint=0.90,
                    cost=1.5,
                    metadata={"view": "abaxial_surface"}
                ),
                "stem_view": EvidenceItem(
                    evidence_type="stem_view",
                    modality="image",
                    data_path_or_content=e3_path,
                    reliability_hint=0.85,
                    cost=1.8,
                    metadata={"view": "petiole_stem"}
                ),
                "symptom_text": EvidenceItem(
                    evidence_type="symptom_text",
                    modality="text",
                    data_path_or_content=d_info["text"],
                    reliability_hint=0.98,
                    cost=0.5,
                    metadata={"source": "farmer_survey"}
                ),
                "weather_context": EvidenceItem(
                    evidence_type="weather_context",
                    modality="text",
                    data_path_or_content=d_info["weather"],
                    reliability_hint=0.92,
                    cost=0.4,
                    metadata={"source": "sensor_or_forecast"}
                )
            }

            rec = CaseRecord(
                case_id=case_id,
                plant_id=plant_id,
                crop="tomato",
                disease=disease,
                initial_image_path=init_path,
                initial_md5=init_md5,
                available_evidence=ev_items,
                source_dataset="EvidenceGain_Tomato_Benchmark",
                license="CC-BY-4.0"
            )
            cases.append(rec)

    manifest = DatasetManifest(
        version="1.0.0",
        crop="tomato",
        classes=classes,
        cases=cases,
        created_at="2026-09-14",
        metadata={"num_classes": len(classes), "total_cases": len(cases)}
    )

    manifest_path = "data/manifests/dataset_manifest.json"
    manifest.save_json(manifest_path)
    logger.info(f"Dataset manifest generated successfully with {len(cases)} cases at {manifest_path}")
    return manifest

if __name__ == "__main__":
    create_synthetic_benchmark_dataset()
