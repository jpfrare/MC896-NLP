"""Métricas formais de avaliação para entidades e relações (restrito às 5 classes válidas)."""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from . import config

def compute_relation_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> Dict[str, Any]:
    """Calcula métricas globais e por classe para extração de relações.
    
    A avaliação é restrita às 5 classes de config.REAL_RELATIONS.
    Caso a predição seja SEM_PREDICAO ou uma classe fora do escopo, ela é tratada
    como incorreta (penalizando o recall da classe verdadeira).
    """
    labels = config.REAL_RELATIONS
    
    # Substituir valores nulos ou inesperados por SEM_PREDICAO
    y_pred_clean = y_pred.fillna(config.NO_PRED).astype(str)
    
    report = classification_report(
        y_true,
        y_pred_clean,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred_clean, labels=labels)
    
    metrics = {
        "model": model_name,
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"]
            }
            for cls in labels
        },
        "confusion_matrix": cm,
        "labels": labels
    }
    
    return metrics

def compute_entity_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> Dict[str, Any]:
    """Calcula métricas formais para reconhecimento e classificação de entidades."""
    labels = config.REAL_ENTITY_TYPES
    y_pred_clean = y_pred.fillna(config.NO_PRED).astype(str)
    
    report = classification_report(
        y_true,
        y_pred_clean,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    
    return {
        "model": model_name,
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"]
            }
            for cls in labels if cls in report
        }
    }