"""Gold standard unificado do Projeto 2.

Le as anotacoes manuais (data/annotations/text*.csv) e produz um unico
DataFrame com o esquema comum que todos os modelos usam:

    doc_id, e1_text, e1_type, e2_text, e2_type, context, relation
"""
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT / "data" / "annotations"            
ANNOTATIONS_TEST = PROJECT / "data" / "annotations_test"  
GOLD_PATH = PROJECT / "data" / "processed" / "gold.csv"
GOLD_TEST_PATH = PROJECT / "data" / "processed" / "gold_test.csv"

# Vocabulario padrao
NODE_TYPES = ["PATIENT", "SYMPTOM", "PROCEDURE", "FINDING", "DIAGNOSIS", "VALUE"]

LABELS = [
    "HAS_SYMPTOM",
    "UNDERWENT_PROCEDURE",
    "SUPPORTS",
    "TREATED_BY",
    "HAS_VALUE",
    "NO_RELATION",
]
COLUMNS = ["doc_id", "e1_text", "e1_type", "e2_text", "e2_type", "context", "relation"]


def build_gold(annotations_dir=ANNOTATIONS):
    """Junta text*.csv e valida. Falha alto se houver rotulo vazio ou desconhecido."""
    files = sorted(Path(annotations_dir).glob("text*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum text*.csv em {annotations_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={"intermediate_text": "context", "relation_label": "relation"})
    df["context"] = df["context"].fillna("").astype(str).str.strip()

    missing = df[COLUMNS].isna().sum()
    if missing.any():
        raise ValueError(f"Valores vazios no gold:\n{missing[missing > 0]}")
    unknown = set(df["relation"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Rotulos fora de LABELS: {sorted(unknown)}")
    bad_types = (set(df["e1_type"]) | set(df["e2_type"])) - set(NODE_TYPES)
    if bad_types:
        raise ValueError(f"Tipos de no fora de NODE_TYPES: {sorted(bad_types)}")

    return df[COLUMNS].reset_index(drop=True)


def build_entities(gold):
    """Entidades unicas (doc_id, text, type) do gold: dados do classificador de nos (etapa 1)."""
    a = gold[["doc_id", "e1_text", "e1_type"]].set_axis(["doc_id", "text", "type"], axis=1)
    b = gold[["doc_id", "e2_text", "e2_type"]].set_axis(["doc_id", "text", "type"], axis=1)
    return pd.concat([a, b]).drop_duplicates().reset_index(drop=True)


def save_gold(split="train"):
    """split='train' le data/annotations; split='test' le data/annotations_test."""
    src, path = (ANNOTATIONS, GOLD_PATH) if split == "train" else (ANNOTATIONS_TEST, GOLD_TEST_PATH)
    df = build_gold(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_gold(split="train"):
    """Carrega o gold (regenera a partir das anotacoes). O teste nao pode ter doc_id do treino."""
    df = save_gold(split)
    if split == "test":
        overlap = set(df["doc_id"]) & set(build_gold(ANNOTATIONS)["doc_id"])
        if overlap:
            raise ValueError(f"Vazamento: textos no treino e no teste: {sorted(overlap)}")
    return df


if __name__ == "__main__":
    for split in ("train", "test"):
        try:
            g = save_gold(split)
        except FileNotFoundError as err:
            print(f"[{split}] ainda nao existe: {err}")
            continue
        print(f"[{split}] {len(g)} pares, {g.doc_id.nunique()} textos")
        print(g.relation.value_counts().to_string())