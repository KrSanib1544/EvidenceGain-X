# EvidenceGain-X

> **Counterfactual, Self-Verifying Next-Best Evidence Acquisition for Crop Disease Diagnosis**

## Project Overview

Conventional crop disease diagnosis systems ask: *"What disease is this?"*
**EvidenceGain-X** asks two fundamental questions:
1. *"What is the current provisional diagnosis and what are the competing hypotheses?"*
2. *"What observation should the model acquire next to maximize diagnostic discrimination while minimizing cost and risk?"*

### Key Features
- **Disease-Hypothesis-Conditioned Evidence Utility**: Dynamically evaluates candidate observations against competing disease rivals.
- **Counterfactual Evidence Planning**: Estimates expected diagnostic shifts before observation.
- **Reliability & Gating**: Quality-checks acquired evidence (blur, lighting, framing, relevance).
- **Contradiction Verification**: Detects unexpected evidence shifts to prevent false confidence.
- **Adaptive Stopping**: Halts observation when uncertainty reaches a reliable threshold.

## Repository Structure

```text
EvidenceGain-X/
├── configs/          # Master configurations
├── data/             # Raw data, manifests, and processed episodes
├── src/              # Core source package
│   ├── data/         # Loaders, preprocessors, episode builders
│   ├── models/       # Vision backbones & classifiers
│   ├── diagnosis/    # Top-k hypotheses & confusion analysis
│   ├── evidence/     # Evidence definitions & utility scoring
│   ├── planning/     # Counterfactual planning
│   ├── verification/ # Quality gating & contradiction checks
│   ├── evaluation/   # Benchmarking against Baselines B0-B6
│   └── utils/        # Seed, config, logging utilities
├── experiments/      # Experiment tracking & logs
├── checkpoints/      # Model weights
├── results/          # Metric plots and tables
├── demo/             # Interactive Streamlit demo
└── tests/            # Automated test suite
```

## Setup & Execution

```bash
# Create virtual environment
uv venv .venv
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```
