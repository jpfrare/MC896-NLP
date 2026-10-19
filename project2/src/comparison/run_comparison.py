"""Executa TODA a comparação Projeto 1 (regras) x Projeto 2 (N-gramas + ML).

Uso (a partir de project2/):
    python -m src.comparison.run_comparison            # tudo
    python -m src.comparison.run_comparison --no-mlp   # sem PyTorch
    python -m src.comparison.run_comparison --no-extras --seed 7
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import config, data, graph_analysis as ga, metrics, models, p1_baseline, plots

KEY_MODELS = ["Projeto 1 (regras)", "LR n-gramas [manual]", "LR n-gramas [manual, tipos previstos]",
              "MLP TF-IDF [weak sup.]", "LR n-gramas [weak sup.]"]


def df_to_md(df: pd.DataFrame, floatfmt="{:.3f}", index=True) -> str:
    d = df.reset_index() if index else df
    if index and d.columns[0] == "index":
        d = d.rename(columns={"index": "modelo"})
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(out_tables: Path = config.OUT_TABLES, out_figs: Path = config.OUT_FIGS, include_mlp=True,
        include_extras=True, seed=config.SEED, n_boot=1000, log=print) -> dict:
    out_tables, out_figs = Path(out_tables), Path(out_figs)
    out_tables.mkdir(parents=True, exist_ok=True)
    gold = data.load_gold()
    docs = sorted(gold["doc_id"].unique())
    pos_labels = sorted(l for l in gold["relation_label"].unique() if l != config.NO_REL)
    labels = [config.NO_REL] + pos_labels
    log(f"[1/6] Gold: {len(gold)} pares em {len(docs)} documentos {docs}")

    # ---------------- RELAÇÕES ----------------
    log("[2/6] Modelos de relação")
    preds, info, notes = models.relation_predictions(gold, include_mlp, include_extras, seed, log)
    uniform = pd.concat([gold, preds.add_prefix("pred__")], axis=1)
    uniform.to_csv(out_tables / "predicoes_relacoes.csv", index=False)

    rows, cms, pcs, ci = {}, {}, {}, {}
    for name in preds.columns:
        rows[name] = metrics.summarize(gold["relation_label"], preds[name], labels, pos_labels)
        cms[name] = metrics.confusion(gold["relation_label"], preds[name], labels)
        pcs[name] = metrics.per_class(gold["relation_label"], preds[name], labels).set_index("classe")
        ci[name] = metrics.bootstrap_ci(gold["relation_label"], preds[name], labels, n=n_boot, seed=seed)
    summary = pd.DataFrame(rows).T
    for c in summary.columns:
        if c != "never_predicted":
            summary[c] = summary[c].astype(float)
    summary["macro_f1_ic95_inf"] = [ci[n][0] for n in summary.index]
    summary["macro_f1_ic95_sup"] = [ci[n][1] for n in summary.index]
    summary.insert(0, "familia", [info[n][0] for n in summary.index])
    summary.insert(1, "supervisao", [info[n][1] for n in summary.index])
    summary = summary.sort_values("macro_f1", ascending=False)
    summary.to_csv(out_tables / "relacoes_resumo.csv")
    per_class_f1 = pd.DataFrame({n: pcs[n]["f1"] for n in summary.index}).T[labels]
    per_class_f1.to_csv(out_tables / "relacoes_f1_por_classe.csv")
    pd.concat({n: pcs[n] for n in summary.index}).to_csv(out_tables / "relacoes_por_classe_completo.csv")
    pd.concat({n: cms[n] for n in summary.index}).to_csv(out_tables / "relacoes_matrizes_confusao.csv")

    # ---------------- ABLAÇÃO DE N-GRAMAS ----------------
    log("[3/6] Ablação de n-gramas")
    abl = models.ngram_ablation(gold, seed, log)
    abl_rows = {}
    for (gran, setting), pred in abl.items():
        m = metrics.summarize(gold["relation_label"], pred, labels, pos_labels)
        key = "com_tipos" if setting == "com tipos" else "so_texto"
        abl_rows.setdefault(gran, {}).update({f"macro_f1_{key}": m["macro_f1"], f"detect_f1_{key}": m["detect_f1"],
                                              f"accuracy_{key}": m["accuracy"]})
    abl_tbl = pd.DataFrame(abl_rows).T.astype(float)
    abl_tbl.index.name = "granularidade"
    abl_tbl.to_csv(out_tables / "ablacao_ngramas.csv")

    # ---------------- ENTIDADES ----------------
    log("[4/6] Entidades")
    ents = data.gold_entities(gold)
    ent_preds, coverage = models.entity_predictions(ents, seed, log)
    pd.concat([ents, ent_preds.add_prefix("pred__")], axis=1).to_csv(out_tables / "predicoes_entidades.csv", index=False)
    ent_labels = sorted(set(ents["type"]))
    ent_pos = [l for l in ent_labels if l != config.NOT_ENT]
    ent_sum, ent_pc = {}, {}
    for n in ent_preds.columns:
        yt, yp = ents["type"].to_numpy(), ent_preds[n].to_numpy()
        from sklearn.metrics import f1_score, precision_recall_fscore_support
        p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=ent_pos, average="micro", zero_division=0)
        ent_sum[n] = {"accuracy": float((yt == yp).mean()),
                      "macro_f1": float(f1_score(yt, yp, labels=ent_labels, average="macro", zero_division=0)),
                      "tipos_micro_precisao": float(p), "tipos_micro_recall": float(r), "tipos_micro_f1": float(f)}
        ent_pc[n] = metrics.per_class(yt, yp, ent_labels).set_index("classe")["f1"]
    ent_sum = pd.DataFrame(ent_sum).T.sort_values("macro_f1", ascending=False)
    ent_pc = pd.DataFrame(ent_pc)
    ent_sum.to_csv(out_tables / "entidades_resumo.csv")
    ent_pc.to_csv(out_tables / "entidades_f1_por_tipo.csv")

    # ---------------- GRAFO ----------------
    log("[5/6] Análise dos grafos")
    gtable, graphs = ga.graph_comparison_table(gold, preds)
    gtable.to_csv(out_tables / "grafos_comparacao.csv", index=False)
    e2e = p1_baseline.p1_end_to_end(gold, data.load_p1_edges())
    e2e.to_csv(out_tables / "p1_ponta_a_ponta.csv", index=False)
    p1_full = data.load_p1_edges()["relation_label"].value_counts().rename("arestas_P1_corpus_inteiro")
    p1_full.to_csv(out_tables / "p1_relacoes_corpus_inteiro.csv")

    # ---------------- FIGURAS ----------------
    log("[6/6] Gráficos")
    figs = [
        plots.fig_model_ranking(summary, ci, out_figs),
        plots.fig_precision_recall(summary, out_figs),
        plots.fig_per_class_heatmap(per_class_f1, out_figs),
        plots.fig_confusions(cms, labels, out_figs, KEY_MODELS + ["Naive Bayes [manual]", "SVM linear [manual]"]),
        plots.fig_ngram_ablation(abl_tbl, out_figs),
        plots.fig_entities(ent_pc.drop(index=[l for l in ent_pc.index if l not in ent_labels]), ent_sum, out_figs),
        plots.fig_relation_distribution(graphs, out_figs),
        plots.fig_schema_graphs(graphs, ga.schema_counts, out_figs, ["Gold (anotação manual)"] + KEY_MODELS),
    ]

    report = build_report(gold, docs, summary, per_class_f1, abl_tbl, ent_sum, coverage, gtable, e2e, p1_full,
                          notes, seed, figs, out_figs)
    (out_tables / "comparison_report.md").write_text(report, encoding="utf-8")
    (out_tables / "run_info.json").write_text(json.dumps({"seed": seed, "notes": notes, "docs_gold": docs}, indent=2, ensure_ascii=False))
    log(f"\nPronto. Tabelas/relatório: {out_tables}\n        Figuras:          {out_figs}")
    return {"summary": summary, "entities": ent_sum, "graphs": gtable, "ablation": abl_tbl, "figures": figs}


def build_report(gold, docs, summary, pcf1, abl, ent, coverage, gtable, e2e, p1_full, notes, seed, figs, out_figs):
    best = summary.index[0]
    p1 = summary.loc["Projeto 1 (regras)"]
    ml = summary.drop(index=["Projeto 1 (regras)", "Baseline (classe majoritária)"], errors="ignore")
    best_ml = ml.index[0] if len(ml) else None
    obs = []
    if best_ml:
        obs.append(f"Melhor macro-F1 de relações: **{best}** ({summary.loc[best, 'macro_f1']:.3f}); "
                   f"Projeto 1 = {p1['macro_f1']:.3f}; melhor modelo de ML = **{best_ml}** ({ml.loc[best_ml, 'macro_f1']:.3f}).")
    obs.append(f"Projeto 1 detecta {p1['detect_recall']:.0%} dos pares realmente relacionados no gold "
               f"(recall de detecção), com precisão de detecção {p1['detect_precision']:.0%}.")
    for n in summary.index:
        if summary.loc[n, "never_predicted"] != "-" and n != "Baseline (classe majoritária)":
            obs.append(f"`{n}` nunca prevê: {summary.loc[n, 'never_predicted']}.")
    wk, mn = "LR n-gramas [weak sup.]", "LR n-gramas [manual]"
    if wk in summary.index and mn in summary.index:
        d = summary.loc[mn, "macro_f1"] - summary.loc[wk, "macro_f1"]
        obs.append(f"Mesmo modelo (LR + n-gramas), fonte de rótulos diferente: anotação manual {'supera' if d > 0 else 'não supera'} "
                   f"a weak supervision em {abs(d):.3f} de macro-F1 — isola o efeito da supervisão.")
    for col, desc in (("macro_f1_so_texto", "só texto"), ("macro_f1_com_tipos", "texto + tipos do gold")):
        spread = abl[col].max() - abl[col].min()
        if spread < 0.01:
            obs.append(f"Ablação de n-gramas ({desc}): praticamente sem diferença entre granularidades (amplitude {spread:.3f}).")
        else:
            obs.append(f"Ablação de n-gramas ({desc}): melhor = **{abl[col].idxmax()}** ({abl[col].max():.3f}), pior = {abl[col].idxmin()} ({abl[col].min():.3f}).")
    tp, orc = "LR n-gramas [manual, tipos previstos]", "LR n-gramas [manual]"
    if tp in summary.index and orc in summary.index:
        obs.append(f"Sem o oráculo de tipos (tipos previstos pelo EntityClassifier), o LR cai de {summary.loc[orc, 'macro_f1']:.3f} para "
                   f"{summary.loc[tp, 'macro_f1']:.3f} de macro-F1 — comparação mais justa com o Projeto 1, que não recebe tipos prontos.")
    obs.append(f"Entidades: cobertura do Projeto 1 sobre as menções reais do gold = {coverage:.0%} (casamento relaxado).")
    obs.append("Intervalos de confiança largos (gold pequeno): diferenças menores que a largura do IC **não** são conclusivas.")

    rel_cols = ["familia", "supervisao", "accuracy", "macro_f1", "macro_f1_ic95_inf", "macro_f1_ic95_sup", "weighted_f1",
                "pos_micro_precision", "pos_micro_recall", "pos_micro_f1", "detect_f1", "typing_accuracy_on_gold_positives"]
    gcols = ["metodo", "arestas", "nos", "grau_medio", "densidade_media", "arestas_corretas (tipo+par)",
             "arestas_em_excesso", "arestas_perdidas", "jaccard_esquema_vs_gold"]
    rel_path = lambda p: f"../../../assets/images/comparison/{Path(p).name}"
    md = [
        "# Comparação Projeto 1 (regras) × Projeto 2 (N-gramas + ML)", "",
        f"_Gerado automaticamente por `python -m src.comparison.run_comparison` (seed={seed})._", "",
        "## 1. Configuração experimental", "",
        f"- **Teste (gold)**: {len(gold)} pares candidatos em {len(docs)} documentos ({', '.join(docs)}), nunca usados no treino manual.",
        "- **Treino manual**: `data/annotations/text01..12.csv`. **Weak supervision**: arestas do Projeto 1 (`data/processed/edges.csv`) sem os documentos gold.",
        f"- **Lematizador usado**: `{notes.get('lemmatizer')}`. **MLP**: {notes.get('mlp', 'não executado')}.",
        "- Tarefa de relações avaliada sobre os **pares candidatos do gold**. Os modelos marcados `[manual]` recebem os **tipos de entidade do gold (oráculo)**; a variante `tipos previstos` usa o EntityClassifier. O Projeto 1 é lido da sua saída oficial, com casamento relaxado de menções (favorece o P1) e **sem** receber tipos prontos.",
        "- Pares com uma entidade `NOT_ENTITY` são sempre `NO_RELATION`: com tipos do gold isso é fácil para o ML (inflaciona a classe `NO_RELATION`).", "",
        "## 2. Observações automáticas", ""] + [f"- {o}" for o in obs] + ["",
        "## 3. Relações — resumo por modelo", "", df_to_md(summary[rel_cols]), "",
        "## 4. Relações — F1 por classe", "", df_to_md(pcf1), "",
        "## 5. Ablação de n-gramas (Regressão Logística, anotação manual)", "", df_to_md(abl), "",
        "## 6. Entidades (classificação do tipo da menção)", "", df_to_md(ent), "",
        "## 7. Grafo resultante (mesmos pares candidatos)", "", df_to_md(gtable, index=False), "",
        "### Projeto 1 ponta a ponta (todas as arestas do P1 nos documentos gold, mundo fechado)", "",
        df_to_md(e2e, index=False), "",
        "_O gold não é exaustivo: a precisão ponta a ponta tende a ser subestimada._", "",
        "### Arestas do Projeto 1 no corpus inteiro, por relação", "", df_to_md(p1_full.to_frame()), "",
        "## 8. Figuras", ""] + [f"![{Path(f).stem}]({rel_path(f)})" for f in figs]
    return "\n".join(md) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-mlp", action="store_true", help="não treina o MLP (sem PyTorch)")
    ap.add_argument("--no-extras", action="store_true", help="não treina Naive Bayes / SVM / Random Forest")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-tables", type=Path, default=config.OUT_TABLES)
    ap.add_argument("--out-figs", type=Path, default=config.OUT_FIGS)
    a = ap.parse_args(argv)
    run(a.out_tables, a.out_figs, not a.no_mlp, not a.no_extras, a.seed, a.bootstrap)


if __name__ == "__main__":
    main()
