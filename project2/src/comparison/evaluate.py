"""Estágio 3: métricas das três visões a partir dos arquivos pred_*.csv e ent_*.csv.

Visão 1 - tipagem de entidade dado o span (P1 palavras-chave x LR n-gramas), mesmos spans do gold.
Visão 2 - relação dado o par (TIPOS, P1_TABELA, LR, MLP), tipos do gold.
          Tabela principal só nas linhas positivas; NO_RELATION predito conta como erro
          (aparece como coluna extra na matriz de confusão). Falso alarme medido nas
          linhas NO_RELATION do gold.
Visão 3 - ponta a ponta (P1, LR): cobertura, acerto de rótulo/tipos, precisão (limite
          inferior se o gold não for exaustivo), razão |predito|/|gold|, estrutura do grafo.

Saídas: data/processed/comparison/tables/*.csv e assets/images/*.png
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import plots
from .common import (ENTITY_TYPES, NO_REL, NOT_ENT, RELATIONS, load_entity_predictions,
                     load_predictions, ordered, save_table)
from .gold import load_gold, positive_mask
from .match import match_edges, norm_tokens


def prf_table(y_true, y_pred, labels, model):
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = [dict(modelo=model, classe=c, precisao=p[i], recall=r[i], f1=f[i], n=int(n[i]))
            for i, c in enumerate(labels)]
    rows.append(dict(modelo=model, classe="macro", precisao=p.mean(), recall=r.mean(), f1=f.mean(), n=int(n.sum())))
    rows.append(dict(modelo=model, classe="acurácia", precisao=np.nan, recall=np.nan,
                     f1=accuracy_score(y_true, y_pred), n=len(y_true)))
    return rows


# ---------------------------------------------------------------- visão 1
def eval_entities():
    ents = load_entity_predictions()
    if ents.empty:
        print("[entidades] nenhum ent_*.csv; pulando.")
        return
    ents = ents[ents["gold_type"] != NOT_ENT]
    rows, per_model = [], {}
    for model in ordered(ents["model"].unique()):
        d = ents[ents["model"] == model]
        rows += prf_table(d["gold_type"], d["pred_type"], ENTITY_TYPES, model)
        per_model[model] = d
    tab = pd.DataFrame(rows)
    save_table(tab, "entidades_prf")

    n = ents.drop_duplicates("ent_id")["gold_type"].value_counts()
    fig, ax = plots.plt.subplots(figsize=(9, 3.6))
    series = {m: [tab[(tab.modelo == m) & (tab.classe == c)]["f1"].iloc[0] for c in ENTITY_TYPES] for m in per_model}
    plots.grouped_bars(ax, [f"{c}\n(n={n.get(c, 0)})" for c in ENTITY_TYPES], series)
    ax.set_ylabel("F1")
    ax.set_title("Visão 1 — tipagem de entidade dado o span (mesmos spans do gold)", pad=28)
    plots.save(fig, "v1_entidades_f1")

    labels = ENTITY_TYPES + [t for t in [NOT_ENT] if any((d["pred_type"] == t).any() for d in per_model.values())]
    fig, axes = plots.plt.subplots(1, len(per_model), figsize=(4.2 * len(per_model), 3.8))
    for ax, (m, d) in zip(np.atleast_1d(axes), per_model.items()):
        plots.confusion(ax, confusion_matrix(d["gold_type"], d["pred_type"], labels=labels)[: len(ENTITY_TYPES)],
                        ENTITY_TYPES, labels, m)
    np.atleast_1d(axes)[0].set_ylabel("gold")
    plots.save(fig, "v1_entidades_confusao")


# ---------------------------------------------------------------- visão 2
def eval_pairs(pred):
    gold = load_gold().set_index("gold_row")
    pos = positive_mask(gold)
    par = pred[pred["view"] == "par"]
    rows, confs, alarms, per_case = [], {}, [], []
    for model in ordered(par["model"].unique()):
        d = par[par["model"] == model].set_index("gold_row").loc[gold.index]
        yt, yp = gold.loc[pos, "relation_label"], d.loc[pos, "relation_label"]
        rows += prf_table(yt, yp, RELATIONS, model)
        confs[model] = confusion_matrix(yt, yp, labels=RELATIONS + [NO_REL])[: len(RELATIONS)]
        neg = gold["relation_label"] == NO_REL
        alarms.append(dict(modelo=model, n_negativos=int(neg.sum()),
                           falsos_alarmes=int((d.loc[neg, "relation_label"] != NO_REL).sum()),
                           taxa=(d.loc[neg, "relation_label"] != NO_REL).mean(),
                           prediz_no_relation=bool((d["relation_label"] == NO_REL).any())))
        for case_id, idx in gold[pos].groupby("case_id").groups.items():
            per_case.append(dict(modelo=model, caso=case_id, n=len(idx),
                                 acuracia=accuracy_score(gold.loc[idx, "relation_label"], d.loc[idx, "relation_label"])))
    tab = pd.DataFrame(rows)
    save_table(tab, "par_prf")
    save_table(pd.DataFrame(alarms), "par_falsos_alarmes")
    save_table(pd.DataFrame(per_case).pivot(index="modelo", columns="caso", values="acuracia")
               .reindex(ordered(par["model"].unique())).reset_index(), "par_acuracia_por_caso")

    n = gold.loc[pos, "relation_label"].value_counts()
    models = list(confs)
    fig, ax = plots.plt.subplots(figsize=(10, 3.8))
    series = {m: [tab[(tab.modelo == m) & (tab.classe == c)]["f1"].iloc[0] for c in RELATIONS] for m in models}
    plots.grouped_bars(ax, [f"{c}\n(n={n.get(c, 0)})" for c in RELATIONS], series)
    ax.set_ylabel("F1")
    ax.set_title("Visão 2 — relação dado o par (tipos do gold), linhas positivas", pad=28)
    plots.save(fig, "v2_par_f1")

    fig, axes = plots.plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        plots.confusion(ax, confs[m], RELATIONS, RELATIONS + [NO_REL], m)
    np.atleast_1d(axes)[0].set_ylabel("gold")
    plots.save(fig, "v2_par_confusao")

    al = pd.DataFrame(alarms)
    fig, ax = plots.plt.subplots(figsize=(6, 3))
    plots.grouped_bars(ax, [f"pares NO_RELATION do gold (n={al.n_negativos.iloc[0]})"],
                       {r.modelo: [r.taxa] for r in al.itertuples()})
    ax.set_ylabel("taxa de falso alarme")
    ax.set_title("Visão 2 — arestas criadas onde o gold diz que não há relação", pad=28)
    plots.save(fig, "v2_falsos_alarmes")


# ---------------------------------------------------------------- visão 3
def graph_stats(df, model):
    rows = []
    for case_id, d in df.groupby("case_id"):
        g = nx.Graph()
        for r in d.itertuples():
            a, b = norm_tokens(r.e1_text, r.e1_type), norm_tokens(r.e2_text, r.e2_type)
            g.add_edge(" ".join(sorted(a)), " ".join(sorted(b)))
        rows.append(dict(modelo=model, caso=case_id, nos=g.number_of_nodes(), arestas=len(d),
                         grau_medio=2 * g.number_of_edges() / max(g.number_of_nodes(), 1),
                         componentes=nx.number_connected_components(g) if len(g) else 0,
                         pct_arestas_paciente=((d["e1_type"] == "PATIENT") | (d["e2_type"] == "PATIENT")).mean(),
                         pct_negadas=d["negated"].mean() if "negated" in d else np.nan))
    return rows


STATUS_ORDER = [("TP_OK", "certa", "#1A7F37"), ("TP_ROTULO", "rótulo errado", "#D4A017"),
                ("FN", "perdida (só no gold)", "#9E9E9E"), ("FP", "a mais (não está no gold)", "#C62828")]


def status_by_relation(m: pd.DataFrame):
    """Figura estática enxuta do 'excesso x perda': por modelo e relação, quantas arestas do gold
    foram acertadas, acertadas com rótulo errado ou perdidas, e quantas arestas a mais o modelo criou."""
    m = m.copy()
    m["relacao"] = m["gold_label"].where(m["status"] != "FP", m["pred_label"])
    tab = m.groupby(["modelo", "relacao", "status"]).size().unstack(fill_value=0)
    save_table(tab.reset_index(), "e2e_status_por_relacao")
    models = ordered(m["modelo"].unique())
    fig, axes = plots.plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.4), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        total = max([tab.loc[(model, r)].sum() for r in RELATIONS if (model, r) in tab.index] + [1])  # maior barra
        left = np.zeros(len(RELATIONS))
        for status, label, color in STATUS_ORDER:
            vals = np.array([tab.loc[(model, r), status] if (model, r) in tab.index and status in tab else 0
                             for r in RELATIONS], dtype=float)
            ax.barh(RELATIONS, vals, left=left, color=color, edgecolor="white", linewidth=0.8, label=label, height=0.65)
            for y, (lft, v) in enumerate(zip(left, vals)):
                if v >= 3 and v >= 0.04 * total:  # só rotula segmentos largos o bastante
                    ax.text(lft + v / 2, y, int(v), ha="center", va="center", fontsize=7,
                            color="white" if status != "FN" else "#222222")
            left += vals
        ax.set_title(model, fontsize=10)
        ax.grid(axis="x", color="#E6E6E6")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        ax.set_xlabel("nº de arestas")
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(0.5 * len(models), 1.28), ncol=4, fontsize=8)
    fig.suptitle("Visão 3 — por relação: certas, rótulo errado, perdidas e a mais", y=1.12, fontsize=11)
    plots.save(fig, "v3_e2e_status_por_relacao")


def eval_e2e(pred):
    from .common import is_negated
    gold = load_gold(positives_only=True)
    gold["negated"] = gold["intermediate_text"].map(is_negated)
    e2e = pred[pred["view"] == "e2e"]
    summary, per_case, matches, stats = [], [], [], graph_stats(gold, "GOLD")
    counts = [gold["relation_label"].value_counts().rename("GOLD")]
    for model in ordered(e2e["model"].unique()):
        p = e2e[e2e["model"] == model].reset_index(drop=True)
        m = match_edges(gold, p)
        m.insert(0, "modelo", model)
        matches.append(m)
        stats += graph_stats(p, model)
        counts.append(p["relation_label"].value_counts().rename(model))
        for case_id, d in [("TOTAL", m)] + list(m.groupby("case_id")):
            tp = d["status"].isin(["TP_OK", "TP_ROTULO"])
            n_gold = int((d["status"] != "FP").sum())
            n_pred = int((d["status"] != "FN").sum())
            row = dict(modelo=model, caso=case_id, n_gold=n_gold, n_pred=n_pred, casadas=int(tp.sum()),
                       cobertura=tp.sum() / max(n_gold, 1),
                       acerto_rotulo=(d["status"] == "TP_OK").sum() / max(tp.sum(), 1) if tp.any() else np.nan,
                       acerto_tipos=d.loc[tp, "types_ok"].mean() if tp.any() else np.nan,
                       acerto_ponta_a_ponta=(d["status"] == "TP_OK").sum() / max(n_gold, 1),
                       precisao_lim_inf=tp.sum() / max(n_pred, 1),
                       razao_pred_gold=n_pred / max(n_gold, 1))
            (summary if case_id == "TOTAL" else per_case).append(row)
    if not summary:
        print("[e2e] nenhuma predição ponta a ponta; pulando.")
        return
    save_table(pd.DataFrame(summary), "e2e_resumo")
    save_table(pd.DataFrame(per_case), "e2e_por_caso")
    save_table(pd.concat(matches, ignore_index=True), "e2e_casamentos")
    st = pd.DataFrame(stats)
    agg = st.groupby("modelo").agg(nos=("nos", "sum"), arestas=("arestas", "sum"), grau_medio=("grau_medio", "mean"),
                                   componentes=("componentes", "sum"),
                                   pct_arestas_paciente=("pct_arestas_paciente", "mean"),
                                   pct_negadas=("pct_negadas", "mean"))
    save_table(agg.reindex(ordered(agg.index)).reset_index(), "e2e_estrutura_grafo")
    cnt = pd.concat(counts, axis=1).reindex(RELATIONS).fillna(0).astype(int)
    save_table(cnt.reset_index(names="relacao"), "e2e_contagem_relacoes")

    sm = pd.DataFrame(summary)
    metrics = ["cobertura", "acerto_rotulo", "acerto_tipos", "precisao_lim_inf"]
    fig, ax = plots.plt.subplots(figsize=(9, 3.6))
    plots.grouped_bars(ax, ["cobertura\n(recall de pares)", "acerto de rótulo\n(entre casadas)",
                            "acerto de tipos\n(entre casadas)", "precisão\n(limite inferior)"],
                       {r.modelo: [getattr(r, k) for k in metrics] for r in sm.itertuples()})
    ax.set_title(f"Visão 3 — ponta a ponta nos 3 casos do gold (n_gold={sm.n_gold.iloc[0]})", pad=28)
    plots.save(fig, "v3_e2e_metricas")

    status_by_relation(pd.concat(matches, ignore_index=True))

    fig, ax = plots.plt.subplots(figsize=(10, 3.6))
    plots.grouped_bars(ax, RELATIONS, {c: cnt[c].tolist() for c in cnt.columns}, ylim=None, fmt="{:.0f}")
    ax.set_ylabel("nº de arestas")
    ax.set_title("Visão 3 — arestas por tipo de relação (excesso x perda)", pad=28)
    plots.save(fig, "v3_e2e_contagem_relacoes")


def main():
    pred = load_predictions()
    print("Visão 1: entidades")
    eval_entities()
    print("Visão 2: pares")
    eval_pairs(pred)
    print("Visão 3: ponta a ponta")
    eval_e2e(pred)


if __name__ == "__main__":
    main()
