"""Avaliacao compartilhada: todo modelo do grupo deve ser medido por aqui.

Uso tipico:

    from gold import load_gold
    from evaluation import leave_one_doc_out, print_report

    def fit_predict(train_df, test_df):
        ...                      # treina em train_df, devolve lista de rotulos para test_df
        return preds

    res = leave_one_doc_out(load_gold(), fit_predict)
    print_report(res, "meu_modelo")
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from gold import LABELS


def evaluate(y_true, y_pred, labels=LABELS):
    """P/R/F1 por classe, macro-F1, acuracia e matriz de confusao.

    Levanta erro se algum rotulo for vazio ou desconhecido, para nunca
    transformar lixo em uma classe sem avisar.
    """
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Tamanhos diferentes: {len(y_true)} vs {len(y_pred)}")
    bad = (set(y_true) | set(y_pred)) - set(labels)
    if bad:
        raise ValueError(f"Rotulos desconhecidos ou vazios: {sorted(map(str, bad))}")

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=labels
    )
    present = per_class["support"] > 0  # macro so sobre classes que existem no gold
    return {
        "per_class": per_class,
        "macro_f1": float(per_class.loc[present, "f1"].mean()),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "confusion": pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=[f"true:{l}" for l in labels],
            columns=[f"pred:{l}" for l in labels],
        ),
        "n": len(y_true),
    }


def leave_one_doc_out(gold, fit_predict, labels=LABELS, target="relation"):
    """Validacao cruzada deixando um texto (doc_id) de fora por vez.

    Evita vazamento entre trechos do mesmo caso clinico. `fit_predict(train, test)`
    recebe dois DataFrames do gold e devolve uma lista de rotulos para `test`.
    As predicoes dos folds sao juntadas e avaliadas de uma vez; `fold_macro_f1`
    mostra a variacao entre textos. Para o classificador de nos (etapa 1) use
    `target="type", labels=NODE_TYPES` sobre `build_entities(gold)`.
    """
    y_true, y_pred, fold_scores = [], [], {}
    for doc in sorted(gold["doc_id"].unique()):
        test = gold[gold["doc_id"] == doc].reset_index(drop=True)
        train = gold[gold["doc_id"] != doc].reset_index(drop=True)
        preds = list(fit_predict(train, test))
        if len(preds) != len(test):
            raise ValueError(f"fit_predict devolveu {len(preds)} rotulos para {len(test)} pares")
        fold_scores[doc] = evaluate(test[target], preds, labels)["macro_f1"]
        y_true += list(test[target])
        y_pred += preds

    res = evaluate(y_true, y_pred, labels)
    res["fold_macro_f1"] = fold_scores
    res["y_true"], res["y_pred"] = y_true, y_pred
    return res


def train_test(train, test, fit_predict, labels=LABELS, target="relation"):
    """Treina em `train` e mede em `test` (conjunto separado, nunca visto no treino).

    Esta e a comparacao principal entre modelos. `fit_predict` tem a mesma forma de sempre.
    Para um modelo que ja esta treinado, faca fit_predict ignorar `train`.
    """
    overlap = set(train["doc_id"]) & set(test["doc_id"])
    if overlap:
        raise ValueError(f"Vazamento: textos no treino e no teste: {sorted(overlap)}")
    preds = list(fit_predict(train, test))
    if len(preds) != len(test):
        raise ValueError(f"fit_predict devolveu {len(preds)} rotulos para {len(test)} pares")
    res = evaluate(test[target], preds, labels)
    res["y_true"], res["y_pred"] = list(test[target]), preds
    return res


def compare_table(results, labels=LABELS):
    """Tabela final: uma linha por modelo, com accuracy, macro-F1 e F1 de cada classe.
    `results` e um dict {nome: resultado de evaluate/train_test/leave_one_doc_out}."""
    rows = []
    for name, res in results.items():
        row = {"modelo": name, "accuracy": round(res["accuracy"], 3),
               "macro_f1": round(res["macro_f1"], 3)}
        for lab in labels:
            row[f"f1_{lab}"] = round(float(res["per_class"].loc[lab, "f1"]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion(res, path):
    """Matriz de confusao em CSV (cole no Excel/Sheets ou plote para o README)."""
    res["confusion"].to_csv(path)


def print_report(res, name="modelo"):
    print(f"\n=== {name} (n={res['n']}) ===")
    print(f"macro-F1: {res['macro_f1']:.3f} | accuracy: {res['accuracy']:.3f}")
    if "fold_macro_f1" in res:
        folds = res["fold_macro_f1"]
        vals = np.array(list(folds.values()))
        print(f"macro-F1 por texto: media {vals.mean():.3f}, desvio {vals.std():.3f}")
    print(res["per_class"].round(3).to_string())
    print(res["confusion"].to_string())


def summary_row(res, name):
    """Uma linha para a tabela final que vai no README."""
    row = {"modelo": name, "macro_f1": round(res["macro_f1"], 3),
           "accuracy": round(res["accuracy"], 3)}
    if "fold_macro_f1" in res:
        row["desvio_entre_textos"] = round(float(np.std(list(res["fold_macro_f1"].values()))), 3)
    return row  