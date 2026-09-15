import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import json

from src.utils.config import load_config
from src.data.manifest import DatasetManifest
from src.models.classifier import DiseaseClassifier
from src.evidence.selector import EvidenceGainXPolicy
from src.diagnosis.pipeline import EvidenceGainXPipeline

st.set_page_config(
    page_title="EvidenceGain-X Diagnostic System",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 EvidenceGain-X: Active Next-Best Evidence Crop Diagnostic System")
st.markdown(
    "**Counterfactual, Self-Verifying Next-Best Evidence Acquisition for Crop Disease Diagnosis**"
)

@st.cache_resource
def get_pipeline():
    cfg = load_config("configs/config.yaml")
    classes = cfg["dataset"]["classes"]
    cand_types = cfg["evidence"]["candidate_types"]
    
    classifier = DiseaseClassifier(num_classes=len(classes), pretrained=False)
    policy = EvidenceGainXPolicy(classes=classes, candidate_types=cand_types)
    pipeline = EvidenceGainXPipeline(
        classifier=classifier,
        policy=policy,
        classes=classes,
        max_budget=cfg["evidence"]["max_budget"]
    )
    return cfg, pipeline, classes

cfg, pipeline, classes = get_pipeline()

manifest_path = "data/manifests/dataset_manifest.json"
if Path(manifest_path).exists():
    manifest = DatasetManifest.load_json(manifest_path)
    case_ids = [c.case_id for c in manifest.cases]
else:
    manifest = None
    case_ids = []

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Initial Crop Observation")
    selected_case_id = st.selectbox("Select Benchmark Case:", case_ids) if case_ids else None
    
    if selected_case_id and manifest:
        case = next(c for c in manifest.cases if c.case_id == selected_case_id)
        img = Image.open(case.initial_image_path)
        st.image(img, caption=f"Initial Observation ({case.crop})", use_container_width=True)
        
        st.markdown(f"**True Ground Truth (Hidden from model):** `{case.disease}`")
        
        if st.button("Run Diagnostic Inquiry Loop", type="primary"):
            st.session_state["run_case"] = case

with col_right:
    if "run_case" in st.session_state:
        case = st.session_state["run_case"]
        st.subheader("2. Diagnostic Inquiry & Evidence Trail")
        
        # Run episode
        res = pipeline.run_episode(case.initial_image_path, case.available_candidates)
        
        # Provisional stage
        st.info(f"**Provisional Diagnosis:** {res['leading_disease']} (Confidence: {res['leading_prob']*100:.1f}%)")
        if res.get("rival_disease"):
            st.warning(f"**Competing Hypothesis Conflict:** `{res['leading_disease']}` ↔ `{res['rival_disease']}` (Margin: {res['margin']:.2f})")
        
        st.write("---")
        st.subheader("3. Sequential Evidence Acquisition Steps")
        
        for item in res["evidence_trail"]:
            with st.expander(f"Step {item['step']}: Acquired '{item['evidence_type']}'", expanded=True):
                st.write(f"- **Quality Action:** `{item['action']}` (Score: {item['quality_score']:.2f})")
                st.write(f"- **Contradiction Detected:** `{item.get('contradiction', False)}` (Score: {item.get('contradiction_score', 0.0):.2f})")
                st.write(f"- **Updated Hypothesis:** `{item['leading_after']}` (Prob: {item['leading_prob_after']*100:.1f}%)")
                st.write(f"- **Shannon Entropy Remaining:** `{item['entropy_after']:.3f}`")

        st.write("---")
        st.subheader("4. Final Verified Diagnosis")
        st.success(
            f"**Final Diagnosis:** {res['final_diagnosis']} | "
            f"**Confidence:** {res['final_confidence']*100:.1f}% | "
            f"**Steps Taken:** {res['total_steps_taken']}"
        )
