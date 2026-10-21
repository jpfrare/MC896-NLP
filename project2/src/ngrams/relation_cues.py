"""Nomes compostos de relação (Possibilidade 3 do guia): quais n-gramas do texto intermediário
estão associados a cada rótulo, aprendidos dos 12 textos anotados, sem regra escrita à mão.

Score = PMI(n-grama, rótulo) sobre as linhas anotadas (presença do n-grama na linha),
com contagem mínima no rótulo. N-gramas formados só por stopwords ("to be", "the first")
são descartados, assim como n-gramas que só estendem uma pista já escolhida ("also denied"
depois de "denied"). Empates são desfeitos de forma determinística (contagem, depois ordem
alfabética): sem isso o resultado mudava a cada execução, pela aleatoriedade do hash de str.
Depois verifica no gold quantas arestas positivas contêm pelo menos uma das pistas do seu
rótulo (cobertura das pistas).

Saídas: tables/ngramas_pistas_relacao.csv, tables/ngramas_pistas_cobertura_gold.csv
"""
import math
from collections import Counter

import pandas as pd

from src.comparison.common import NO_REL, RELATIONS, save_table
from src.comparison.gold import load_gold, load_training_annotations
from src.ngrams.text import STOP, tokenize

MIN_COUNT = 3
TOP = 6


def grams(text, nmax=3) -> set:
    toks = [t for t in tokenize(text) if t[0].isalnum()]
    return {" ".join(toks[i:i + n]) for n in range(1, nmax + 1) for i in range(len(toks) - n + 1)}


def main():
    train = load_training_annotations()
    train["grams"] = train["intermediate_text"].fillna("").map(grams)
    n_rows = len(train)
    label_n = train["relation_label"].value_counts()
    gram_n = Counter(g for gs in train["grams"] for g in gs)
    rows = []
    for label in RELATIONS + [NO_REL]:
        sub = train[train["relation_label"] == label]
        joint = Counter(g for gs in sub["grams"] for g in gs)
        for g, c in sorted(joint.items()):
            if c < MIN_COUNT or all(t in STOP for t in g.split()):
                continue
            pmi = math.log2((c / n_rows) / ((gram_n[g] / n_rows) * (label_n[label] / n_rows)))
            rows.append(dict(relacao=label, pista=g, n_no_rotulo=c, n_total=gram_n[g],
                             p_rotulo_dado_pista=c / gram_n[g], pmi=pmi))
    ranked = pd.DataFrame(rows).sort_values(["relacao", "pmi", "n_no_rotulo", "pista"],
                                            ascending=[True, False, False, True], kind="mergesort")
    keep = []
    for _, d in ranked.groupby("relacao", sort=True):
        chosen = []
        for r in d.itertuples():
            if any(f" {c} " in f" {r.pista} " for c in chosen):
                continue  # só estende uma pista já escolhida
            chosen.append(r.pista)
            keep.append(r.Index)
            if len(chosen) == TOP:
                break
    cues = ranked.loc[keep].reset_index(drop=True)
    save_table(cues, "ngramas_pistas_relacao")

    gold = load_gold(positives_only=True)
    cue_sets = cues.groupby("relacao")["pista"].apply(set).to_dict()
    gold["tem_pista"] = [bool(grams(t) & cue_sets.get(lab, set()))
                         for t, lab in zip(gold["intermediate_text"].fillna(""), gold["relation_label"])]
    cov = gold.groupby("relation_label")["tem_pista"].agg(n="size", com_pista="sum").reset_index()
    cov["fracao"] = cov["com_pista"] / cov["n"]
    save_table(cov, "ngramas_pistas_cobertura_gold")
    print(cues.groupby("relacao")["pista"].apply(lambda s: " | ".join(s)).to_string())
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
