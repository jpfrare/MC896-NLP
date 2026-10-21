"""Desenha o Modelo Lógico do grafo de propriedades (assets/images/modelo-logico-grafos.png).

Tipos de nó e de aresta seguem o esquema de anotação (gold e textos de treino):
  PATIENT -UNDERWENT_PROCEDURE-> PROCEDURE
  PATIENT -HAS_SYMPTOM-> {SYMPTOM, FINDING, DIAGNOSIS}
  PROCEDURE -SUPPORTS-> {SYMPTOM, FINDING, DIAGNOSIS}
  {SYMPTOM, FINDING, DIAGNOSIS} -TREATED_BY-> PROCEDURE
  {PROCEDURE, SYMPTOM, FINDING, DIAGNOSIS} -HAS_VALUE-> VALUE
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.comparison.common import FIGURES, ROOT, TYPE_COLORS  # noqa: E402

NODE_PROPS = "text · normalizado · case_id"
EDGE_PROPS = "relation_label · intermediate_text · negated · model"


def node(ax, xy, name, w=0.17, h=0.085):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="white", ec=TYPE_COLORS[name], lw=2.2))
    ax.text(x, y + 0.012, name, ha="center", va="center", fontsize=11, weight="bold", color="#222222")
    ax.text(x, y - 0.022, NODE_PROPS, ha="center", va="center", fontsize=6.5, color="#666666")


def arrow(ax, a, b, label, rad=0.0, lxy=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.5, color="#444444",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4))
    mx, my = lxy if lxy else ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ax.text(mx, my, label, ha="center", va="center", fontsize=8.5, color="#0B3D91",
            bbox=dict(fc="white", ec="none", pad=1.5))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.70, 0.20), 0.27, 0.58, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#F7F7F7", ec="#999999", ls="--", lw=1.2))
    ax.text(0.835, 0.755, "achado clínico", ha="center", fontsize=9, color="#555555", style="italic")
    pos = {"PATIENT": (0.12, 0.50), "PROCEDURE": (0.47, 0.84), "VALUE": (0.47, 0.12),
           "SYMPTOM": (0.835, 0.64), "FINDING": (0.835, 0.49), "DIAGNOSIS": (0.835, 0.34)}
    for n, xy in pos.items():
        node(ax, xy, n)
    arrow(ax, (0.17, 0.55), (0.40, 0.80), "UNDERWENT_PROCEDURE", rad=-0.15, lxy=(0.24, 0.72))
    arrow(ax, (0.205, 0.50), (0.70, 0.50), "HAS_SYMPTOM", lxy=(0.32, 0.515))
    arrow(ax, (0.555, 0.84), (0.72, 0.70), "SUPPORTS", rad=-0.2, lxy=(0.66, 0.83))
    arrow(ax, (0.72, 0.60), (0.53, 0.795), "TREATED_BY", rad=-0.25, lxy=(0.58, 0.66))
    arrow(ax, (0.72, 0.28), (0.555, 0.13), "HAS_VALUE", rad=-0.15, lxy=(0.66, 0.17))
    arrow(ax, (0.47, 0.795), (0.47, 0.165), "HAS_VALUE", lxy=(0.47, 0.32))
    ax.text(0.5, 0.985, "Modelo lógico — grafo de propriedades do conhecimento clínico", ha="center",
            va="top", fontsize=13, weight="bold")
    ax.text(0.5, 0.025, f"Propriedades das arestas: {EDGE_PROPS}.  Setas que tocam a caixa 'achado clínico' "
            "valem para SYMPTOM, FINDING e DIAGNOSIS.", ha="center", fontsize=8, color="#444444")
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "modelo-logico-grafos.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=170)
    plt.close(fig)
    print(f"  figura -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
