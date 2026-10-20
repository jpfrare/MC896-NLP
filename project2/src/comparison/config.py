"""Caminhos e constantes do sistema de comparação (tudo relativo à pasta project2/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations"
PROCESSED = DATA / "processed"
NOTEBOOKS = ROOT / "pipelines" / "notebooks"

MLP_CHECKPOINT = PROCESSED / "mlp_relation_pipeline_completo.pt"
OUT_TABLES = PROCESSED / "comparison"
OUT_FIGS = ROOT / "assets" / "images" / "comparison"

SEED = 42
NO_REL = "NO_RELATION"
NOT_ENT = "NOT_ENTITY"
NO_PRED = "SEM_PREDICAO"

REAL_RELATIONS = ["HAS_SYMPTOM", "HAS_VALUE", "SUPPORTS", "TREATED_BY", "UNDERWENT_PROCEDURE"]
REAL_ENTITY_TYPES = ["PATIENT", "SYMPTOM", "FINDING", "DIAGNOSIS", "PROCEDURE", "VALUE"]
E2E_THRESHOLD = 0.5

MODEL_COLORS = {
    'Projeto 1 (Regras)': '#1f77b4',
    'Pipeline 1': '#2ca02c',
    'Pipeline 2': '#ff7f0e'
}