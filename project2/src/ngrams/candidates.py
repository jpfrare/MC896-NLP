"""Experimento ousado e barato: colocações por PMI como GERADOR DE CANDIDATOS a entidade.

Pergunta: segmentar o texto por associação estatística entre palavras (NPMI) recupera as
entidades compostas do gold tão bem quanto os noun chunks do parser (usados pelo LR)?

Geradores, aplicados aos 3 casos do gold:
  conteudo     sequências máximas de palavras de conteúdo (corta em stopword/pontuação) - baseline
  pmi          as mesmas sequências, cortadas onde NPMI(bigrama) < LIMIAR (estatísticas do corpus todo);
               o LIMIAR é escolhido nos 12 textos de treino (maior recall estrito de entidades compostas)
  noun_chunks  sintagmas nominais do spaCy (MedicalTextProcessor do pipeline2) - requer en_core_sci_sm
  uniao        noun_chunks ∪ pmi
Avaliação: recall das entidades do gold (SYMPTOM, FINDING, DIAGNOSIS, PROCEDURE), estrito
(mesmo conjunto de tokens) e relaxado (Jaccard >= 0,5), separado para entidades de 1 palavra
e compostas, e o nº de candidatos (custo, proxy de precisão).

Saídas: tables/ngramas_candidatos.csv, assets/images/ngramas_candidatos.png
"""
import sys

import pandas as pd

from src.comparison import plots
from src.comparison.common import GOLD_IDS, PIPELINE2, RAW, save_table
from src.comparison.gold import load_clinical_entities
from src.comparison.match import norm_tokens
from src.ngrams.text import NgramStats, is_content, sentences

THRESHOLDS = (0.2, 0.4, 0.6, 0.8)   # o limiar principal é escolhido nos 12 textos de TREINO, nunca no gold
GEN_COLORS = {"conteudo": "#999999", "pmi": "#E69F00", "noun_chunks": "#0072B2", "uniao": "#CC79A7"}


def content_runs(text) -> list:
    runs = []
    for toks in sentences(text):
        cur = []
        for t in toks:
            if is_content(t) or (cur and t[0].isdigit()):
                cur.append(t)
            else:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
    return runs


def pmi_chunks(text, stats: NgramStats, threshold: float) -> list:
    out = []
    for run in content_runs(text):
        cur = [run[0]]
        for a, b in zip(run, run[1:]):
            if stats.npmi((a, b)) < threshold:
                out.append(cur)
                cur = []
            cur.append(b)
        out.append(cur)
    return out


def noun_chunks(text):
    try:
        sys.path.insert(0, str(PIPELINE2))
        from useful.preprocess import MedicalTextProcessor
        proc = MedicalTextProcessor()
        return [c["text"] for s in proc.split_sentences(text) for c in proc.extract_entity_candidates(proc.nlp(s))]
    except Exception as exc:  # spaCy/en_core_sci_sm ausente
        print(f"[candidatos] noun_chunks indisponível ({type(exc).__name__}: {exc}); pulando.")
        return None


def evaluate(cands_by_case: dict, ents: pd.DataFrame, name: str) -> list:
    rows = []
    for multi in (False, True):
        sub = ents[ents["multi"] == multi]
        strict = relaxed = 0
        for r in sub.itertuples():
            cs = cands_by_case[r.case_id]
            strict += r.toks in cs
            relaxed += any(len(r.toks & c) / len(r.toks | c) >= 0.5 for c in cs)
        rows.append(dict(gerador=name, entidades="compostas" if multi else "1 palavra", n=len(sub),
                         recall_estrito=strict / max(len(sub), 1), recall_relaxado=relaxed / max(len(sub), 1),
                         candidatos=sum(len(v) for v in cands_by_case.values())))
    return rows


def as_sets(chunks):
    return {frozenset(norm_tokens(" ".join(c) if isinstance(c, list) else c)) - {""} for c in chunks} - {frozenset()}


def select_threshold(cases: pd.DataFrame, stats: NgramStats) -> tuple:
    """Escolhe o limiar de NPMI nos 12 textos de treino do LR (não toca no gold)."""
    ents = load_clinical_entities("treino")
    texts = cases.set_index("case_id").loc[sorted(ents["case_id"].unique()), "case_text"].to_dict()
    rows = []
    for th in THRESHOLDS:
        g = {c: as_sets(pmi_chunks(t, stats, th)) for c, t in texts.items()}
        r = [x for x in evaluate(g, ents, f"pmi@{th}") if x["entidades"] == "compostas"][0]
        rows.append(dict(limiar=th, recall_estrito_compostas_treino=r["recall_estrito"], candidatos=r["candidatos"]))
    sel = pd.DataFrame(rows).sort_values(["recall_estrito_compostas_treino", "candidatos"], ascending=[False, True])
    save_table(sel.sort_values("limiar"), "ngramas_candidatos_selecao_limiar")
    return float(sel["limiar"].iloc[0]), sel


def main():
    cases = pd.read_csv(RAW / "cases.csv")
    stats = NgramStats(cases["case_text"])
    texts = cases.set_index("case_id").loc[GOLD_IDS, "case_text"].to_dict()
    main_th, sel = select_threshold(cases, stats)
    print(f"[candidatos] limiar de NPMI escolhido no treino: {main_th}")
    print(sel.sort_values("limiar").round(3).to_string(index=False))

    ents = load_clinical_entities("gold")

    gens = {"conteudo": {c: as_sets(content_runs(t)) for c, t in texts.items()}}
    rows = evaluate(gens["conteudo"], ents, "conteudo")
    for th in THRESHOLDS:
        g = {c: as_sets(pmi_chunks(t, stats, th)) for c, t in texts.items()}
        name = "pmi" if th == main_th else f"pmi@{th}"
        gens[name] = g
        rows += evaluate(g, ents, name)
    nc = {}
    for c, t in texts.items():
        nc[c] = noun_chunks(t)
        if nc[c] is None:
            break
    if all(v is not None for v in nc.values()):
        gens["noun_chunks"] = {c: as_sets(v) for c, v in nc.items()}
        gens["uniao"] = {c: gens["noun_chunks"][c] | gens["pmi"][c] for c in texts}
        rows += evaluate(gens["noun_chunks"], ents, "noun_chunks") + evaluate(gens["uniao"], ents, "uniao")
    tab = pd.DataFrame(rows)
    save_table(tab, "ngramas_candidatos")
    print(tab.round(3).to_string(index=False))

    main_gens = [g for g in ["conteudo", "pmi", "noun_chunks", "uniao"] if g in gens]
    cats = [(e, m) for e in ["1 palavra", "compostas"] for m in ["recall_estrito", "recall_relaxado"]]
    labels = [f"{m.split('_')[1]}\n{e} (n={tab[(tab.entidades == e)].n.iloc[0]})" for e, m in cats]
    fig, ax = plots.plt.subplots(figsize=(10, 3.8))
    import numpy as np
    x = np.arange(len(cats))
    w = 0.8 / len(main_gens)
    for i, g in enumerate(main_gens):
        vals = [tab[(tab.gerador == g) & (tab.entidades == e)][m].iloc[0] for e, m in cats]
        n_c = tab[tab.gerador == g].candidatos.iloc[0]
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, w * 0.92, color=GEN_COLORS[g], label=f"{g} ({n_c} candidatos)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall de entidades do gold")
    ax.legend(ncol=len(main_gens), loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
    ax.set_title(f"Geradores de candidatos a entidade nos casos do gold (pmi: NPMI ≥ {main_th}, limiar escolhido no treino)", pad=30, fontsize=10)
    plots.save(fig, "ngramas_candidatos")


if __name__ == "__main__":
    main()
