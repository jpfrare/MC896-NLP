"""Adaptador do Projeto 1 (regras).

Gera três saídas:
  pred_P1_e2e.csv        arestas que as regras extraíram nos casos do gold (data/processed/edges.csv)
  ent_P1.csv             tipagem dos spans do gold pelos dicionários de palavras-chave do P1
  pred_P1_TABELA_par.csv as regras do P1 vistas como classificador "dado o par": o rótulo que o P1
                         atribui a cada par de tipos (maioria em edges.csv, fora dos casos do gold).
                         Pares com NOT_ENTITY -> NO_RELATION (o P1 descarta não-entidades).
Também salva a tabela usada em p1_tabela_tipos.csv (vai para o README).
"""
import pandas as pd

from .common import GOLD_IDS, NO_REL, NOT_ENT, OUT, PROCESSED, save_entity_predictions, save_predictions, to_schema
from .gold import load_gold, load_gold_entities
from .p1_typing import p1_type


def p1_fallback(e1_type: str, e2_type: str) -> str:
    """Espelha a lógica das regras do P1 para pares de tipos que não aparecem em edges.csv."""
    if NOT_ENT in (e1_type, e2_type):
        return NO_REL
    if e2_type == "VALUE":
        return "HAS_VALUE"
    if e1_type == "PATIENT":
        return "HAS_SYMPTOM" if e2_type == "SYMPTOM" else "TREATED_BY"
    return "SUPPORTS"


def main():
    edges = pd.read_csv(PROCESSED / "edges.csv")

    # --- visão ponta a ponta -------------------------------------------------
    e2e = edges[edges["case_id"].isin(GOLD_IDS)]
    save_predictions(to_schema(e2e, "P1", "e2e"), "P1", "e2e")

    # --- visão "dado o par": tabela tipo->rótulo das regras ------------------
    fora_gold = edges[~edges["case_id"].isin(GOLD_IDS)]
    table = (fora_gold.groupby(["e1_type", "e2_type"])["relation_label"]
             .agg(lambda s: s.value_counts().index[0]).rename("relation_label").reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "p1_tabela_tipos.csv", index=False)
    lookup = {(a, b): r for a, b, r in table.itertuples(index=False)}

    gold = load_gold()
    pair = gold.copy()
    pair["relation_label"] = [
        NO_REL if NOT_ENT in (a, b) else lookup.get((a, b), p1_fallback(a, b))
        for a, b in zip(gold["e1_type"], gold["e2_type"])
    ]
    save_predictions(to_schema(pair, "P1_TABELA", "par"), "P1_TABELA", "par")

    # --- tipagem de entidades dado o span (mesmos spans para P1 e LR) --------
    ents = load_gold_entities()
    ents["pred_type"] = ents["text"].map(p1_type)
    save_entity_predictions(ents, "P1")


if __name__ == "__main__":
    main()
