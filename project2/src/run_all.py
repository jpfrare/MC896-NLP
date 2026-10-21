"""Roda todo o sistema de comparação, na ordem, a partir de project2/:

    python -m src.run_all            # tudo
    python -m src.run_all --rapido   # pula a ablação (a parte mais lenta)

Antes de começar, apaga as saídas da execução anterior (previsões, tabelas, figuras geradas).
Cada etapa é independente: se uma falhar (p.ex. torch ou en_core_sci_sm ausentes),
as demais continuam e o resumo final diz o que foi pulado e por quê.
Normalmente é chamado por `python3 run.py`, que também cria o ambiente e abre o painel.
"""
import argparse
import importlib
import shutil
import time
import traceback

from src.comparison.common import FIGURES, HTML, OUT

GENERATED_FIGURES = ("v1_*.png", "v2_*.png", "v3_*.png", "ngramas_*.png", "lr_*.png", "grafo_*.png",
                     "modelo-logico-grafos.png")


def clean_outputs():
    """Apaga as saídas da execução anterior, para que uma etapa que falhe agora não deixe
    previsões antigas entrarem nas métricas."""
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.rmtree(HTML, ignore_errors=True)
    for pattern in GENERATED_FIGURES:
        for f in FIGURES.glob(pattern):
            f.unlink()


STEPS = [
    ("ambiente", "src.check_env", "Python 3.12, NumPy < 2, spaCy 3.7 e modelos spaCy"),
    ("gold", "src.comparison.gold", "carrega e resume o gold"),
    ("P1", "src.comparison.adapt_p1", "adaptador do Projeto 1 (e2e, tabela de tipos, tipagem)"),
    ("TIPOS", "src.comparison.baselines", "baseline só-tipos"),
    ("LR", "src.comparison.run_lr", "Regressão Logística (par, entidades, e2e)"),
    ("MLP", "src.comparison.infer_mlp", "inferência do MLP (par)"),
    ("checagem", "src.comparison.check", "sanidade do esquema"),
    ("métricas", "src.comparison.evaluate", "métricas e figuras das três visões"),
    ("colocações", "src.ngrams.collocations", "PMI/NPMI no corpus"),
    ("pistas", "src.ngrams.relation_cues", "n-gramas associados a cada relação"),
    ("candidatos", "src.ngrams.candidates", "PMI como gerador de candidatos"),
    ("LR n-gramas", "src.ngrams.lr_features", "coeficientes e ablação do ngram_range"),
    ("modelo lógico", "src.graph.logical_model", "figura do modelo lógico"),
    ("relatório", "src.report", "tabelas no README e em assets/resultados.md"),
    ("painel", "src.graph.dashboard", "painel interativo assets/html/index.html"),
]
NEEDS_FAILED = {"relatório", "painel"}


def main(rapido: bool = False):
    clean_outputs()
    summary, failed = [], []
    for name, module, desc in STEPS:
        if rapido and name == "LR n-gramas":
            summary.append((name, "pulado (--rapido)", 0.0))
            continue
        print(f"\n===== {name}: {desc}")
        t0 = time.time()
        try:
            mod = importlib.import_module(module)
            mod.main(failed=list(failed)) if name in NEEDS_FAILED else mod.main()
            summary.append((name, "ok", time.time() - t0))
        except Exception as exc:  # segue para a próxima etapa
            traceback.print_exc(limit=2)
            msg = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"
            summary.append((name, f"FALHOU: {msg}", time.time() - t0))
            failed.append((name, msg))
    print("\n===== Resumo")
    for name, status, dt in summary:
        print(f"  {name:14s} {dt:6.1f}s  {status}")
    if failed:
        print("\n  ATENÇÃO: resultados INCOMPLETOS; o README e o painel trazem um aviso listando as etapas que falharam.")
    return failed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    raise SystemExit(1 if main(ap.parse_args().rapido) else 0)
