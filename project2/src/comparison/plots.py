"""Estilo e gráficos padronizados (cor fixa por modelo, n em todos os rótulos de classe)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import FIGURES, MODEL_COLORS, MODEL_HATCH, ROOT  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figura -> {path.relative_to(ROOT)}")
    return path


def grouped_bars(ax, categories, series: dict, ylim=(0, 1.05), value_labels=True, fmt="{:.2f}"):
    """series = {modelo: [valor por categoria]}; cor/hachura fixas por modelo."""
    models = list(series)
    k = len(models)
    width = 0.8 / max(k, 1)
    x = np.arange(len(categories))
    for i, m in enumerate(models):
        vals = np.asarray(series[m], dtype=float)
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, np.nan_to_num(vals), width * 0.92, label=m, color=MODEL_COLORS.get(m, "#555555"),
                      hatch=MODEL_HATCH.get(m), edgecolor="white", linewidth=0.6)
        if value_labels:
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, fmt.format(v),
                            ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=min(k, 6), loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)


def confusion(ax, matrix, rows, cols, title):
    m = np.asarray(matrix, dtype=float)
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("predito")
