"""Métricas de classificação, matriz de confusão e IC por bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import config


def per_class(y_true, y_pred, labels) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"classe": labels, "precisao": p, "recall": r, "f1": f, "suporte": s})


def confusion(y_true, y_pred, labels) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)


def summarize(y_true, y_pred, labels, positive_labels) -> dict:
    """Métricas globais. `positive_labels` = classes que são relação de fato (sem NO_RELATION)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(positive_labels), average="micro", zero_division=0)
    t_bin, p_bin = y_true != config.NO_REL, y_pred != config.NO_REL
    dp, dr, df, _ = precision_recall_fscore_support(t_bin, p_bin, average="binary", zero_division=0)
    typing = float((y_true[t_bin] == y_pred[t_bin]).mean()) if t_bin.any() else 0.0
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "pos_micro_precision": float(mp), "pos_micro_recall": float(mr), "pos_micro_f1": float(mf),
        "detect_precision": float(dp), "detect_recall": float(dr), "detect_f1": float(df),
        "typing_accuracy_on_gold_positives": typing,
        "never_predicted": ", ".join(l for l in labels if l not in set(y_pred)) or "-",
    }


def bootstrap_ci(y_true, y_pred, labels, n=1000, seed=config.SEED, alpha=0.05) -> tuple[float, float]:
    """IC (percentil) do macro-F1 reamostrando os itens do gold com reposição."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))
