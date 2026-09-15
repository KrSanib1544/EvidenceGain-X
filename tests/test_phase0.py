import pytest
from pathlib import Path
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

def test_config_loading():
    cfg = load_config("configs/config.yaml")
    assert cfg is not None
    assert "project" in cfg
    assert cfg["project"]["name"] == "EvidenceGain-X"
    assert "dataset" in cfg
    assert "classifier" in cfg
    assert "evidence" in cfg
    assert "selector" in cfg

def test_seed_reproducibility():
    set_seed(42)
    import random
    import numpy as np
    r1 = random.random()
    n1 = np.random.rand()
    
    set_seed(42)
    r2 = random.random()
    n2 = np.random.rand()
    
    assert r1 == r2
    assert n1 == n2

def test_logger():
    logger = setup_logger("TestLogger")
    assert logger is not None
