"""Análise do GRAFO resultante: dois modelos com F1 parecido podem gerar grafos bem diferentes."""
from collections import Counter

import pandas as pd

from . import config


def positive_edges(gold: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Arestas (pares com relação != NO_RELATION) de um conjunto de rótulos sobre os pares do gold."""
    df = gold[["doc_id", "e1_text", "e1_type", "e2_text", "e2_type"]].copy()
    df["relation"] = labels.values
    return df[df["relation"] != config.NO_REL].reset_index(drop=True)


def graph_stats(edges: pd.DataFrame) -> dict:
    if edges.empty:
        return {"arestas": 0, "nos": 0, "grau_medio": 0.0, "densidade_media": 0.0,
                "tipos_de_relacao": 0, "triplas_de_esquema": 0}
    nodes = set(zip(edges["doc_id"], edges["e1_text"].str.lower())) | set(zip(edges["doc_id"], edges["e2_text"].str.lower()))
    dens = []
    for _, g in edges.groupby("doc_id"):
        n = len(set(g["e1_text"].str.lower()) | set(g["e2_text"].str.lower()))
        dens.append(len(g) / (n * (n - 1)) if n > 1 else 0.0)
    return {"arestas": len(edges), "nos": len(nodes), "grau_medio": 2 * len(edges) / len(nodes),
            "densidade_media": float(sum(dens) / len(dens)), "tipos_de_relacao": edges["relation"].nunique(),
            "triplas_de_esquema": len(schema_counts(edges))}


def schema_counts(edges: pd.DataFrame) -> Counter:
    """Contagem de triplas de esquema (tipo_e1, RELAÇÃO, tipo_e2) — ex.: PATIENT -HAS_SYMPTOM-> SYMPTOM."""
    return Counter(zip(edges["e1_type"], edges["relation"], edges["e2_type"]))


def schema_jaccard(a: Counter, b: Counter) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def edge_set(edges: pd.DataFrame) -> set:
    return set(zip(edges["doc_id"], edges["e1_text"].str.lower(), edges["e2_text"].str.lower(), edges["relation"]))


def graph_comparison_table(gold: pd.DataFrame, relation_preds: pd.DataFrame):
    """Estatísticas do grafo de cada método vs. grafo gold + sobreposição de arestas e de esquema."""
    gold_edges = positive_edges(gold, gold["relation_label"])
    gold_schema, gold_set = schema_counts(gold_edges), edge_set(gold_edges)
    rows, graphs = [], {"Gold (anotação manual)": gold_edges}
    for name in relation_preds.columns:
        graphs[name] = positive_edges(gold, relation_preds[name])
    for name, edges in graphs.items():
        st = graph_stats(edges)
        es = edge_set(edges)
        st["arestas_corretas (tipo+par)"] = len(es & gold_set)
        st["arestas_em_excesso"] = len(es - gold_set)
        st["arestas_perdidas"] = len(gold_set - es)
        st["jaccard_esquema_vs_gold"] = schema_jaccard(schema_counts(edges), gold_schema)
        rows.append({"metodo": name, **st})
    return pd.DataFrame(rows), graphs
