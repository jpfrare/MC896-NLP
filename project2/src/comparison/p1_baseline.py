"""Projeto 1 (regras) como baseline, lido da SAÍDA OFICIAL (nodes.csv / edges.csv).

Cada item do gold é procurado na saída do Projeto 1 com um casamento "relaxado" de menções:
coeficiente de sobreposição de tokens |A∩B| / min(|A|,|B|) >= limiar (padrão 0.5).
Isso FAVORECE o Projeto 1 (aceita casamentos parciais, ex.: "pain" ~ "abdominal pain");
portanto o resultado dele é um teto.

O Projeto 1 não prevê NO_RELATION nem NOT_ENTITY: se nada casa, o item recebe SEM_PREDICAO,
que conta como erro de recall da classe verdadeira.
"""
import re

import pandas as pd

from . import config

_TOK = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> frozenset:
    return frozenset(_TOK.findall(str(text).lower()))


def similarity(a: str, b: str) -> tuple[float, float]:
    """(coef. de sobreposição, Jaccard) entre dois textos."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0, 0.0
    inter = len(ta & tb)
    return inter / min(len(ta), len(tb)), inter / len(ta | tb)


def _best_node(text: str, doc_nodes: pd.DataFrame, thr: float):
    best, best_key = None, (-1.0, -1.0)
    for row in doc_nodes.itertuples(index=False):
        s = max(similarity(text, row.label_original), similarity(text, row.label_normalizado))
        if s[0] >= thr and s > best_key:
            best, best_key = row, s
    return best


def p1_entity_predictions(entities: pd.DataFrame, p1_nodes: pd.DataFrame,
                          thr: float = config.E2E_THRESHOLD) -> pd.Series:
    """Tipo atribuído pelo P1 a cada menção do gold; sem nó correspondente => SEM_PREDICAO."""
    by_doc = {d: g for d, g in p1_nodes.groupby("doc_id")}
    empty = p1_nodes.iloc[0:0]
    preds = []
    for row in entities.itertuples(index=False):
        node = _best_node(row.text, by_doc.get(row.doc_id, empty), thr)
        preds.append(node.node_type if node is not None else config.NO_PRED)
    return pd.Series(preds, index=entities.index)


def _pair_match(e1: str, e2: str, x1: str, x2: str, thr: float):
    """Chave de qualidade do casamento (par e1,e2) x (x1,x2), ou None."""
    s1, s2 = similarity(e1, x1), similarity(e2, x2)
    if min(s1[0], s2[0]) >= thr:
        return (min(s1[0], s2[0]), min(s1[1], s2[1]))
    return None


def _match_edge(e1: str, e2: str, doc_edges: pd.DataFrame, thr: float):
    """Melhor aresta do P1 para o par (e1, e2), aceitando orientação invertida."""
    best, best_key = None, (-1.0, -1.0, -1)
    for row in doc_edges.itertuples(index=False):
        for swapped in (0, 1):
            x1, x2 = (row.e2_text, row.e1_text) if swapped else (row.e1_text, row.e2_text)
            k = _pair_match(e1, e2, x1, x2, thr)
            if k is not None and (k + (-swapped,)) > best_key:
                best, best_key = row, k + (-swapped,)
    return best


def p1_relation_predictions(gold: pd.DataFrame, p1_edges: pd.DataFrame,
                            thr: float = config.E2E_THRESHOLD) -> pd.Series:
    """Rótulo do P1 para cada par do gold; sem aresta correspondente => SEM_PREDICAO."""
    by_doc = {d: g for d, g in p1_edges.groupby("doc_id")}
    empty = p1_edges.iloc[0:0]
    preds = []
    for row in gold.itertuples(index=False):
        edge = _match_edge(row.e1_text, row.e2_text, by_doc.get(row.doc_id, empty), thr)
        preds.append(edge.relation_label if edge is not None else config.NO_PRED)
    return pd.Series(preds, index=gold.index)