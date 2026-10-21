"""Colocações por PMI no corpus (data/raw/cases.csv): entidades compostas e associações entre termos.

Saídas:
  tables/ngramas_colocacoes.csv        bigramas e trigramas de conteúdo (n >= 3) com PMI e NPMI
  tables/ngramas_ranking_precisao.csv  precisão@k de três rankings (frequência, PMI, NPMI):
                                       fração dos top-k que é trecho de alguma entidade anotada
  tables/ngramas_cobertura_gold.csv    fração das entidades compostas do gold que contém uma colocação
  assets/images/ngramas_colocacoes.png

A "verdade" do ranking é o léxico de entidades anotadas manualmente (text*.csv + gold*.csv,
sem NOT_ENTITY). Como só 15 dos 56 casos foram anotados, a precisão@k é um LIMITE INFERIOR.
"""
import pandas as pd

from src.comparison import plots
from src.comparison.common import ANNOTATIONS, NOT_ENT, RAW, save_table
from src.comparison.gold import load_clinical_entities
from src.ngrams.text import NgramStats, tokenize

MIN_COUNT = 3
KS = (10, 20, 50)


def entity_lexicon() -> set:
    frames = [pd.read_csv(p) for p in sorted(ANNOTATIONS.glob("*.csv"))]
    ann = pd.concat(frames, ignore_index=True)
    texts = pd.concat([ann.loc[ann.e1_type != NOT_ENT, "e1_text"], ann.loc[ann.e2_type != NOT_ENT, "e2_text"]])
    return {" ".join(tokenize(t)) for t in texts.dropna().unique()}


def inside_lexicon(ngram: str, lexicon: set) -> bool:
    pad = f" {ngram} "
    return any(pad in f" {e} " for e in lexicon)


def main():
    cases = pd.read_csv(RAW / "cases.csv")
    stats = NgramStats(cases["case_text"])
    lex = entity_lexicon()

    col = pd.DataFrame(stats.collocations(2, MIN_COUNT) + stats.collocations(3, MIN_COUNT))
    col["ordem"] = col["ngram"].str.count(" ") + 1
    col["em_entidade_anotada"] = col["ngram"].map(lambda g: inside_lexicon(g, lex))
    col = col.sort_values("npmi", ascending=False).reset_index(drop=True)
    save_table(col, "ngramas_colocacoes")

    bi = col[col["ordem"] == 2]
    rows = []
    for crit in ["n", "pmi", "npmi"]:
        ranked = bi.sort_values([crit, "ngram"], ascending=[False, True])
        for k in KS:
            top = ranked.head(k)
            rows.append(dict(criterio=crit, k=k, n_bigramas=len(bi), precisao_at_k=top["em_entidade_anotada"].mean()))
    prec = pd.DataFrame(rows)
    save_table(prec, "ngramas_ranking_precisao")

    # mesma definição de "entidade composta" usada em candidates.py (gold.load_clinical_entities)
    multi = load_clinical_entities("gold")
    multi = multi[multi["multi"]].copy()
    multi["toks"] = multi["text"].map(lambda t: [x for x in tokenize(t) if x[0].isalnum()])
    multi["t"] = multi["text"]
    colset = set(col.loc[col["npmi"] > 0, "ngram"])
    multi["contem_colocacao"] = multi["toks"].map(
        lambda ts: any(" ".join(ts[i:i + n]) in colset for n in (2, 3) for i in range(len(ts) - n + 1)))
    cov = pd.DataFrame([dict(entidades_compostas_gold=len(multi), com_colocacao=int(multi["contem_colocacao"].sum()),
                             fracao=multi["contem_colocacao"].mean(),
                             exemplos=", ".join(multi.loc[multi["contem_colocacao"], "t"].head(8)))])
    save_table(cov, "ngramas_cobertura_gold")

    # figura: top-20 bigramas por NPMI, cor indica se o bigrama é trecho de entidade anotada
    top = bi.sort_values(["npmi", "ngram"], ascending=[False, True]).head(20).iloc[::-1]
    fig, (ax1, ax2) = plots.plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = ["#0072B2" if e else "#BBBBBB" for e in top["em_entidade_anotada"]]
    ax1.barh(top["ngram"] + "  (n=" + top["n"].astype(str) + ")", top["npmi"], color=colors, height=0.7)
    ax1.set_xlabel("NPMI")
    ax1.grid(axis="x", color="#E6E6E6")
    ax1.grid(axis="y", visible=False)
    ax1.set_title(f"Top-20 bigramas por NPMI (n ≥ {MIN_COUNT}, {cases.case_id.nunique()} casos)", fontsize=10)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="#0072B2", label="trecho de entidade anotada"),
                        Patch(color="#BBBBBB", label="não aparece no léxico anotado")],
               loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, fontsize=8)
    for crit, style, dy in [("n", ":", 0), ("pmi", "--", 6), ("npmi", "-", -6)]:
        d = prec[prec.criterio == crit]
        ax2.plot(d["k"], d["precisao_at_k"], style, marker="o", color="#0072B2" if crit != "n" else "#999999",
                 label={"n": "frequência", "pmi": "PMI", "npmi": "NPMI"}[crit])
        ax2.annotate({"n": "frequência", "pmi": "PMI", "npmi": "NPMI"}[crit], (d["k"].iloc[-1], d["precisao_at_k"].iloc[-1]),
                     textcoords="offset points", xytext=(5, dy), fontsize=8, va="center")
    ax2.set_xticks(list(KS))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("k")
    ax2.set_ylabel("precisão@k (limite inferior)")
    ax2.set_title("Ranking de bigramas x léxico anotado", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    plots.save(fig, "ngramas_colocacoes")
    print(prec.pivot(index="criterio", columns="k", values="precisao_at_k").round(2).to_string())
    print(cov.drop(columns="exemplos").to_string(index=False))


if __name__ == "__main__":
    main()
