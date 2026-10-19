"""Carregamento uniformizado de TODOS os dados usados na comparação."""
import sys
from pathlib import Path
import pandas as pd
from . import config

PAIR_COLS = ["doc_id", "e1_text", "e1_type", "e2_text", "e2_type", "intermediate_text", "relation_label"]

def _read_annotation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[PAIR_COLS].copy()

def load_manual_annotations() -> list[pd.DataFrame]:
    """text01..text12: anotação manual usada para TREINAR a LR."""
    files = sorted(config.ANNOTATIONS.glob("text*.csv"))
    return [_read_annotation(f) for f in files]

def load_gold() -> tuple[pd.DataFrame, dict]:
    """gold01..gold03 (teste), SEM as linhas NO_RELATION e SEM entidades NOT_ENTITY."""
    files = sorted(config.ANNOTATIONS.glob("gold*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum gold*.csv em {config.ANNOTATIONS}")
    raw = pd.concat([_read_annotation(f) for f in files], ignore_index=True)
    not_ent = (raw["e1_type"] == config.NOT_ENT) | (raw["e2_type"] == config.NOT_ENT)
    no_rel = raw["relation_label"] == config.NO_REL
    gold = raw[~(not_ent | no_rel)].reset_index(drop=True)
    gold.insert(0, "pair_id", range(len(gold)))
    counts = {
        "linhas_brutas": len(raw),
        "removidas_no_relation": int(no_rel.sum()),
        "removidas_not_entity": int(not_ent.sum()),
        "linhas_avaliadas": len(gold)
    }
    return gold, counts

def gold_entities(gold: pd.DataFrame) -> pd.DataFrame:
    """Menções únicas (doc, texto, tipo) do gold filtrado."""
    a = gold[["doc_id", "e1_text", "e1_type"]].set_axis(["doc_id", "text", "type"], axis=1)
    b = gold[["doc_id", "e2_text", "e2_type"]].set_axis(["doc_id", "text", "type"], axis=1)
    ents = pd.concat([a, b]).drop_duplicates().reset_index(drop=True)
    ents.insert(0, "ent_id", range(len(ents)))
    return ents

def load_p1_edges() -> pd.DataFrame:
    """Saída oficial do Projeto 1 (regras): arestas."""
    df = pd.read_csv(config.PROCESSED / "edges.csv", dtype=str, keep_default_na=False)
    return df.rename(columns={"case_id": "doc_id"})

def load_p1_nodes() -> pd.DataFrame:
    """Saída oficial do Projeto 1 (regras): nós."""
    df = pd.read_csv(config.PROCESSED / "nodes.csv", dtype=str, keep_default_na=False)
    return df.rename(columns={"case_id": "doc_id"})

def import_pipeline_code():
    """Importa Train.py do pipeline sem copiá-lo."""
    path = str(config.NOTEBOOKS)
    if path not in sys.path:
        sys.path.insert(0, path)
    from importlib import import_module

    Train = import_module("Train")
    return Train