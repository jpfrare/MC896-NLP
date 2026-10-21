"""N-gramas que a Regressão Logística aprendeu (coef_ + get_feature_names_out) e ablação do ngram_range.

1) Top n-gramas por classe nos dois classificadores (entidade e relação).
2) Peso dos TIPOS x TEXTO no classificador de relação: fração da massa |coef| em cada bloco
   de features, por classe (evidência quantitativa de que a relação é decidida pelos tipos).
3) Ablação (Possibilidade 5 do guia): ngram_range (1,1), (1,2), (1,3), (1,5) no
   EntityClassifier (tipagem dos spans do gold), no RelationClassifier com tipos e no
   RelationClassifier SÓ TEXTO (tipos trocados por constante; com tipos, a relação já sai
   perfeita e a ablação não informaria nada). Para cada configuração, conta quantos n-gramas
   de cada ordem sobrevivem ao vocabulário (min_df=2, max_features=2000 do código original):
   é isso que explica uma curva plana.

Requer spaCy + en_core_sci_sm (lematizador de useful/train.py).
Saídas: tables/lr_top_ngramas.csv, tables/lr_massa_tipos_texto.csv, tables/lr_ablacao_ngram.csv,
        tables/lr_vocabulario_por_ordem.csv,
        assets/images/lr_ablacao_ngram.png, assets/images/lr_massa_tipos_texto.png
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.comparison import plots
from src.comparison.common import ENTITY_TYPES, MODEL_COLORS, NOT_ENT, RELATIONS, save_table
from src.comparison.gold import load_gold, load_gold_entities, positive_mask
from src.comparison.run_lr import NO_TYPE, train_entity, train_relation

TOP = 8
RANGES = [(1, 1), (1, 2), (1, 3), (1, 5)]


def top_features(clf, n_prefix: int, task: str) -> list:
    names = clf.idf_tf.get_feature_names_out()
    rows = []
    for ci, cls in enumerate(clf.model.classes_):
        coefs = clf.model.coef_[ci if clf.model.coef_.shape[0] > 1 else 0][n_prefix:]
        for j in np.argsort(-coefs)[:TOP]:
            rows.append(dict(tarefa=task, classe=cls, ngrama=names[j], ordem=names[j].count(" ") + 1, coef=coefs[j]))
    return rows


def vocab_by_order(clf) -> dict:
    names = clf.idf_tf.get_feature_names_out()
    orders = pd.Series([n.count(" ") + 1 for n in names]).value_counts()
    return {f"{k}-gramas": int(orders.get(k, 0)) for k in range(1, 6)}


def main():
    rel = train_relation()
    ent = train_entity()
    n_types = len(rel.ohe.get_feature_names_out())
    tops = pd.DataFrame(top_features(ent, 0, "entidade") + top_features(rel, n_types, "relacao"))
    save_table(tops, "lr_top_ngramas")

    mass = []
    for ci, cls in enumerate(rel.model.classes_):
        c = np.abs(rel.model.coef_[ci])
        mass.append(dict(classe=cls, massa_tipos=c[:n_types].sum() / c.sum(), n_feats_tipos=n_types,
                         n_feats_texto=len(c) - n_types))
    mass = pd.DataFrame(mass)
    save_table(mass, "lr_massa_tipos_texto")

    fig, ax = plots.plt.subplots(figsize=(8, 3))
    ax.barh(mass["classe"], mass["massa_tipos"], color=MODEL_COLORS["LR"], height=0.6)
    for y, v in enumerate(mass["massa_tipos"]):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#E6E6E6")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"fração de |coef| nas {n_types} features de tipo (o resto: {len(c) - n_types} n-gramas)")
    ax.set_title("LR de relação: quanto do peso está nos tipos das entidades", fontsize=10)
    plots.save(fig, "lr_massa_tipos_texto")

    ents = load_gold_entities()
    ents = ents[ents["gold_type"] != NOT_ENT]
    gold = load_gold()
    pos = gold[positive_mask(gold)]
    rows, vocab = [], []
    const = pd.Series(NO_TYPE, index=pos.index)
    for rng in RANGES:
        label = f"{rng[0]}-{rng[1]}"
        e = train_entity(ngram_range=rng)
        pe = e.predict(ents["text"].astype(str))
        r = train_relation(ngram_range=rng)
        pr = r.predict(pos["e1_type"], pos["e2_type"], pos["intermediate_text"])
        rt = train_relation(text_only=True, ngram_range=rng)
        prt = rt.predict(const, const, pos["intermediate_text"])
        rows.append(dict(ngram_range=label,
                         f1_macro_entidades=f1_score(ents["gold_type"], pe, labels=ENTITY_TYPES, average="macro", zero_division=0),
                         n_entidades=len(ents),
                         f1_macro_relacoes=f1_score(pos["relation_label"], pr, labels=RELATIONS, average="macro", zero_division=0),
                         f1_macro_relacoes_so_texto=f1_score(pos["relation_label"], prt, labels=RELATIONS, average="macro", zero_division=0),
                         n_relacoes=len(pos)))
        vocab.append(dict(ngram_range=label, classificador="entidade", **vocab_by_order(e)))
        vocab.append(dict(ngram_range=label, classificador="relacao", **vocab_by_order(r)))
        vocab.append(dict(ngram_range=label, classificador="relacao_so_texto", **vocab_by_order(rt)))
    save_table(pd.DataFrame(vocab), "lr_vocabulario_por_ordem")
    ab = pd.DataFrame(rows)
    save_table(ab, "lr_ablacao_ngram")
    print(ab.round(3).to_string(index=False))

    fig, ax = plots.plt.subplots(figsize=(6.5, 3.4))
    for col, style, lab in [("f1_macro_entidades", "-", f"tipagem de entidades (n={len(ents)})"),
                            ("f1_macro_relacoes", "--", f"relação, com tipos (n={len(pos)})"),
                            ("f1_macro_relacoes_so_texto", ":", f"relação, só texto (n={len(pos)})")]:
        ax.plot(ab["ngram_range"], ab[col], style, marker="o", color=MODEL_COLORS["LR"], label=lab)
        for x, v in zip(ab["ngram_range"], ab[col]):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("ngram_range do TF-IDF (palavras)")
    ax.set_ylabel("F1 macro")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Ablação de n-gramas na Regressão Logística", fontsize=10)
    plots.save(fig, "lr_ablacao_ngram")


if __name__ == "__main__":
    main()
