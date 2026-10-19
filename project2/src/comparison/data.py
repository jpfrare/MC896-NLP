"""Carregamento uniformizado de TODOS os dados usados na comparação."""
import sys
from pathlib import Path

import pandas as pd

from . import config

PAIR_COLS = ["doc_id", "e1_text", "e1_type", "e2_text", "e2_type", "intermediate_text", "relation_label"]


def _read_annotation(path: Path) -> pd.DataFrame:
    # keep_default_na=False: textos como "NA"/"None" não viram NaN; vazio vira ''
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[PAIR_COLS].copy()


def load_manual_annotations() -> list[pd.DataFrame]:
    """text01..text12: anotação manual usada para TREINAR o ML tradicional."""
    files = sorted(config.ANNOTATIONS.glob("text*.csv"))
    return [_read_annotation(f) for f in files]


def load_gold() -> pd.DataFrame:
    """gold01..gold03: documentos separados para TESTE (nunca vistos no treino manual)."""
    files = sorted(config.ANNOTATIONS.glob("gold*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum gold*.csv em {config.ANNOTATIONS}")
    gold = pd.concat([_read_annotation(f) for f in files], ignore_index=True)
    gold.insert(0, "pair_id", range(len(gold)))
    return gold


def gold_entities(gold: pd.DataFrame) -> pd.DataFrame:
    """Menções únicas (doc, texto, tipo) presentes no gold, incluindo NOT_ENTITY."""
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


def load_weak_training(exclude_docs) -> pd.DataFrame:
    """Weak supervision: arestas das REGRAS do Projeto 1, sem os documentos gold."""
    edges = load_p1_edges()
    edges = edges[~edges["doc_id"].isin(set(exclude_docs))].copy()
    return edges[["doc_id", "e1_type", "e2_type", "intermediate_text", "relation_label"]].reset_index(drop=True)


def import_colleague_code():
    """Importa Train.py (código dos colegas) sem copiá-lo."""
    path = str(config.NOTEBOOKS)
    if path not in sys.path:
        sys.path.insert(0, path)
    import Train  # noqa: E402
    return Train
