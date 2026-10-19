"""Caminhos e constantes do sistema de comparação (tudo relativo à pasta project2/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # .../project2
DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations"
PROCESSED = DATA / "processed"
NOTEBOOKS = ROOT / "pipelines" / "notebooks"           # onde vivem Train.py / Preprocess.py

OUT_TABLES = PROCESSED / "comparison"                  # tabelas, predições e relatório
OUT_FIGS = ROOT / "assets" / "images" / "comparison"   # gráficos (PNG)

SEED = 42
NO_REL = "NO_RELATION"
NOT_ENT = "NOT_ENTITY"
E2E_THRESHOLD = 0.5   # limiar do casamento "relaxado" entre menções (ver p1_baseline.py)
