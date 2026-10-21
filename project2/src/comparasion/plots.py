"""Geração dos gráficos comparativos para a apresentação."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from .config import MODEL_COLORS

def plot_macro_f1(results: list, out_dir: Path):
    df = pd.DataFrame([{"Modelo": r["model"], "Macro-F1": r["macro_f1"]} for r in results])
    df = df.sort_values(by="Macro-F1", ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Macro-F1", y="Modelo", hue="Modelo", legend=False, data=df, palette=MODEL_COLORS)
    plt.title("Ranking: Macro-F1 Geral (5 Classes Válidas)")
    plt.xlim(0, 1.0)
    
    for i, v in enumerate(df["Macro-F1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
        
    plt.tight_layout()
    plt.savefig(out_dir / "01_macro_f1_ranking.png", dpi=300)
    plt.close()

def plot_f1_per_class(results: list, out_dir: Path):
    rows = []
    for r in results:
        for cls, metrics in r["per_class"].items():
            rows.append({"Modelo": r["model"], "Classe": cls, "F1": metrics["f1"]})
            
    df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Classe", y="F1", hue="Modelo", data=df, palette=MODEL_COLORS)
    plt.title("Desempenho (F1-Score) por Classe")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "02_f1_por_classe.png", dpi=300)
    plt.close()

def plot_confusion_matrices(results: list, out_dir: Path):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: 
        axes = [axes]
        
    for ax, res in zip(axes, results):
        sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=res["labels"], yticklabels=res["labels"], ax=ax, cbar=False)
        ax.set_title(res["model"])
        ax.set_ylabel("Verdadeiro")
        ax.set_xlabel("Predito")
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(out_dir / "03_matrizes_confusao.png", dpi=300)
    plt.close()