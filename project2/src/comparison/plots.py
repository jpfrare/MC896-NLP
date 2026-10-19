"""Gráficos (matplotlib, backend sem tela) para a apresentação."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

from . import config

P1 = "Projeto 1 (regras)"


def _color(name):
    if name == P1:
        return "#d62728"
    if "majorit" in name:
        return "#999999"
    if "weak" in name:
        return "#ff7f0e"
    return "#1f77b4"


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_model_ranking(summary, ci, out_dir):
    """Barras de macro-F1 com IC 95% (bootstrap) + F1 de detecção de relação."""
    s = summary.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(s) + 1.6))
    y = np.arange(len(s))
    err = np.array([[m - ci[n][0], ci[n][1] - m] for n, m in zip(s.index, s["macro_f1"])]).T
    ax.barh(y + 0.2, s["macro_f1"], 0.38, xerr=err, color=[_color(n) for n in s.index], capsize=3, label="Macro-F1 (6 classes) ± IC95%")
    ax.barh(y - 0.2, s["detect_f1"], 0.38, color="#bbbbbb", label="F1 de detecção (existe relação?)")
    ax.set_yticks(y); ax.set_yticklabels(s.index); ax.set_xlim(0, 1)
    ax.set_xlabel("F1"); ax.set_title("Classificação de relações no gold (191 pares, 3 documentos)")
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", alpha=0.3)
    return _save(fig, out_dir, "01_ranking_modelos_relacoes.png")


def fig_precision_recall(summary, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n, r in summary.iterrows():
        ax.scatter(r["pos_micro_recall"], r["pos_micro_precision"], s=90, color=_color(n))
        ax.annotate(n, (r["pos_micro_recall"], r["pos_micro_precision"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall (relações positivas, micro)"); ax.set_ylabel("Precisão (relações positivas, micro)")
    ax.set_title("Precisão x Recall nas relações de fato"); ax.grid(alpha=0.3)
    return _save(fig, out_dir, "02_precisao_vs_recall.png")


def fig_per_class_heatmap(per_class_f1, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(per_class_f1) + 1.8))
    im = ax.imshow(per_class_f1.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(per_class_f1.shape[1])); ax.set_xticklabels(per_class_f1.columns, rotation=30, ha="right")
    ax.set_yticks(range(per_class_f1.shape[0])); ax.set_yticklabels(per_class_f1.index)
    for i in range(per_class_f1.shape[0]):
        for j in range(per_class_f1.shape[1]):
            v = per_class_f1.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else "black", fontsize=8)
    ax.set_title("F1 por classe de relação"); fig.colorbar(im, ax=ax, fraction=0.03)
    return _save(fig, out_dir, "03_f1_por_classe.png")


def fig_confusions(cms, labels, out_dir, key_models):
    names = [m for m in key_models if m in cms]
    cols = min(3, len(names)); rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)
    short = [l.replace("UNDERWENT_PROCEDURE", "UNDERWENT").replace("NO_RELATION", "NO_REL") for l in labels]
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n in zip(axes.ravel(), names):
        ax.axis("on")
        cm = cms[n].values.astype(float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("previsto"); ax.set_ylabel("gold"); ax.set_title(n, fontsize=9)
    fig.suptitle("Matrizes de confusão (cor = % da linha; número = contagem)", y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "04_matrizes_confusao.png")


def fig_ngram_ablation(table, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(table)); w = 0.38
    for ax, metric, title in ((axes[0], "macro_f1", "Macro-F1"), (axes[1], "detect_f1", "F1 de detecção (existe relação?)")):
        ax.bar(x - w / 2, table[f"{metric}_com_tipos"], w, label="texto + tipos (gold)", color="#1f77b4")
        ax.bar(x + w / 2, table[f"{metric}_so_texto"], w, label="só texto (n-gramas puros)", color="#ff7f0e")
        ax.set_xticks(x); ax.set_xticklabels(table.index, rotation=20); ax.set_ylim(0, 1)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Granularidade dos n-gramas (Regressão Logística, anotação manual)")
    return _save(fig, out_dir, "05_ablacao_ngramas.png")


def fig_entities(per_class_ent, summary_ent, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cols = list(per_class_ent.columns)
    x = np.arange(len(per_class_ent.index)); w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        ax.bar(x + k * w - 0.4 + w / 2, per_class_ent[c], w, label=f"{c} (macro {summary_ent.loc[c, 'macro_f1']:.2f})", color=_color(c))
    ax.set_xticks(x); ax.set_xticklabels(per_class_ent.index, rotation=20)
    ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.set_title("Classificação de entidades: F1 por tipo")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "06_entidades_f1_por_tipo.png")


def fig_relation_distribution(graphs, out_dir):
    rels = sorted({r for g in graphs.values() for r in g["relation"]})
    names = list(graphs)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / len(names); x = np.arange(len(rels))
    for k, n in enumerate(names):
        counts = [int((graphs[n]["relation"] == r).sum()) for r in rels]
        ax.bar(x + k * w - 0.4 + w / 2, counts, w, label=n, color="black" if n.startswith("Gold") else _color(n), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(rels, rotation=15)
    ax.set_ylabel("nº de arestas"); ax.set_title("Arestas por tipo de relação em cada grafo (mesmos pares candidatos)")
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "07_distribuicao_relacoes_grafo.png")


_REL_COLOR = {"HAS_SYMPTOM": "#d62728", "HAS_VALUE": "#2ca02c", "SUPPORTS": "#1f77b4",
              "UNDERWENT_PROCEDURE": "#9467bd", "TREATED_BY": "#ff7f0e"}


def fig_schema_graphs(graphs, schema_fn, out_dir, key_models):
    """Esquema do grafo no nível de tipos: nós = tipos de entidade, arestas = relações (largura = contagem)."""
    names = [n for n in key_models if n in graphs]
    types = sorted({t for g in graphs.values() for t in list(g["e1_type"]) + list(g["e2_type"])})
    pos = {t: (math.cos(2 * math.pi * i / len(types)), math.sin(2 * math.pi * i / len(types))) for i, t in enumerate(types)}
    cols = min(3, len(names)); rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 6.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n in zip(axes.ravel(), names):
        sc = schema_fn(graphs[n])
        mx = max(sc.values()) if sc else 1
        for t, (x, y) in pos.items():
            ax.scatter([x], [y], s=2700, color="#eef3fb", edgecolor="#555", zorder=2)
            ax.text(x, y, t.replace("PROCEDURE", "PROC."), ha="center", va="center", fontsize=6.5, zorder=3)
        seen = {}
        for (a, rel, b), c in sc.items():
            lw = 0.8 + 5.0 * c / mx
            col = _REL_COLOR.get(rel, "#555")
            if a == b:
                x, y = pos[a]
                ang = math.degrees(math.atan2(y, x))
                ax.add_patch(Arc((x * 1.25, y * 1.25), 0.45, 0.45, theta1=0, theta2=330, color=col, lw=lw, zorder=1))
                ax.text(x * 1.5, y * 1.5, f"{c}", color=col, fontsize=8, ha="center", va="center")
                continue
            k = seen.get(frozenset((a, b)), 0); seen[frozenset((a, b))] = k + 1
            rad = 0.18 + 0.12 * k
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                         mutation_scale=12, lw=lw, color=col, alpha=0.8, shrinkA=22, shrinkB=22, zorder=1))
            mx_, my_ = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
            dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
            ax.text(mx_ - dy * rad * 0.5, my_ + dx * rad * 0.5, str(c), color=col, fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8), zorder=4)
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75); ax.set_aspect("equal"); ax.set_title(f"{n}\n({len(graphs[n])} arestas)", fontsize=10)
    handles = [plt.Line2D([0], [0], color=c, lw=3, label=r) for r, c in _REL_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle("Esquema do grafo extraído (tipo → relação → tipo; número = nº de arestas)", y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return _save(fig, out_dir, "08_esquema_dos_grafos.png")
