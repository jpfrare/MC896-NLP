"""Checagem de sanidade do estágio 2: esquema, contagens e alinhamento com o gold.

Não é a avaliação final (estágio 3). A "prévia" de acurácia serve só para pegar bugs:
o baseline TIPOS tem de dar ~1.0 nas linhas positivas do gold; se não der, o
alinhamento gold_row ou a leitura dos tipos está quebrada.
"""
import pandas as pd

from .common import SCHEMA, load_predictions
from .gold import load_gold, positive_mask


def main():
    pred = load_predictions()
    assert list(pred.columns) == SCHEMA, f"colunas fora do esquema: {list(pred.columns)}"

    print("\n== Linhas por modelo e visão ==")
    print(pd.crosstab(pred["model"], pred["view"]).to_string())

    print("\n== Rótulos preditos por modelo (todas as visões) ==")
    print(pd.crosstab((pred["model"] + "/" + pred["view"]).rename("modelo/visão"),
                      pred["relation_label"]).to_string())

    gold = load_gold()
    pos_ids = set(gold.loc[positive_mask(gold), "gold_row"])
    truth = gold.set_index("gold_row")["relation_label"]

    print("\n== Prévia (visão par, só linhas positivas do gold) ==")
    par = pred[pred["view"] == "par"]
    for model, d in par.groupby("model"):
        assert d["gold_row"].is_unique and set(d["gold_row"]) == set(truth.index), f"{model}: gold_row desalinhado"
        d = d[d["gold_row"].isin(pos_ids)]
        acc = (d["relation_label"].values == truth.loc[d["gold_row"]].values).mean()
        print(f"  {model:10s} n={len(d)}  acurácia={acc:.3f}")

    print("\n== Arestas ponta a ponta por caso ==")
    e2e = pred[pred["view"] == "e2e"]
    print(pd.crosstab(e2e["case_id"], e2e["model"], margins=True).to_string() if len(e2e) else "  (nenhuma)")
    print("\nOK: esquema válido.")


if __name__ == "__main__":
    main()
