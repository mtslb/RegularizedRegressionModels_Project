# src/utils/paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # remonte jusqu'à project_root

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
GRAPHS = ROOT / "graphs"
