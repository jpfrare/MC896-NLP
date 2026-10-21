"""Monta o relatório de resultados a partir das tabelas e injeta no README.md do project2.

- `sections()` define, num só lugar, quais tabelas e figuras compõem o relatório; o README
  (Markdown) e o painel interativo (src/graph/dashboard.py) usam a mesma lista
- escreve assets/resultados.md (todas as tabelas, sempre com n)
- se alguma etapa do run_all falhou, abre a seção com um aviso listando as falhas
- substitui, no README.md, o trecho entre os marcadores
      <!-- RESULTADOS:INICIO -->  e  <!-- RESULTADOS:FIM -->
Nenhum número do README é digitado à mão: todos vêm daqui.
"""
import pandas as pd

from src.comparison.common import FIGURES, RELATIONS, ROOT, TABLES, md_table, ordered

START, END = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIM -->"


def read(name):
    path = TABLES / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def pivot_prf(tab, classes):
    models = ordered(tab["modelo"].unique())
    rows = []
    for c in classes + ["macro", "acurácia"]:
        sub = tab[tab["classe"] == c].set_index("modelo")
        row = {"classe": c, "n": int(sub["n"].iloc[0])}
        row.update({m: sub.loc[m, "f1"] for m in models})
        rows.append(row)
    return pd.DataFrame(rows)


def joined(t, by, col, n=None, code=True):
    fmt = (lambda x: f"`{x}`") if code else str
    return t.groupby(by)[col].apply(lambda s: ", ".join(fmt(x) for x in (s.head(n) if n else s))).reset_index()


# (grupo, título, tabela, transformação, nota, formato numérico)
SPEC = [
    ("Visão 1 — entidades", "Tipagem de entidade dado o span (F1 por tipo)", "entidades_prf",
     lambda t: pivot_prf(t, [c for c in t["classe"].unique() if c not in ("macro", "acurácia")]),
     "P1 = dicionários de palavras-chave; LR = Regressão Logística com n-gramas 1–5. Mesmos spans do gold "
     "(sem NOT_ENTITY). Na linha 'acurácia', a coluna do modelo é a acurácia.", "{:.2f}"),
    ("Visão 2 — relação dado o par", "F1 por classe, tipos do gold, só linhas positivas", "par_prf",
     lambda t: pivot_prf(t, RELATIONS),
     "TIPOS = baseline que só olha (e1_type, e2_type). LR_TEXTO = a mesma LR sem os tipos (comparável ao MLP, "
     "que também só vê texto). Se TIPOS empata com a LR, a relação está sendo decidida pelos tipos.", "{:.2f}"),
    ("Visão 2 — relação dado o par", "Falsos alarmes nos pares NO_RELATION do gold", "par_falsos_alarmes", None,
     "MLP não tem a classe NO_RELATION: toda predição dele é uma aresta.", "{:.2f}"),
    ("Visão 2 — relação dado o par", "Acurácia por caso (variação entre os 3 casos)", "par_acuracia_por_caso", None, "",
     "{:.2f}"),
    ("Visão 3 — ponta a ponta", "P1 e LR nos 3 casos do gold", "e2e_resumo", lambda t: t.drop(columns=["caso"]),
     "cobertura = arestas do gold encontradas; acerto_rotulo/tipos = entre as encontradas; acerto_ponta_a_ponta = "
     "encontradas com o rótulo certo / arestas do gold; precisao_lim_inf é limite inferior se o gold não for "
     "exaustivo; razao_pred_gold > 1 indica excesso de arestas, < 1 indica perda.", "{:.2f}"),
    ("Visão 3 — ponta a ponta", "Por caso", "e2e_por_caso", None, "", "{:.2f}"),
    ("Visão 3 — ponta a ponta", "Estrutura dos grafos (soma/média nos 3 casos)", "e2e_estrutura_grafo", None, "",
     "{:.2f}"),
    ("Visão 3 — ponta a ponta", "Arestas por tipo de relação", "e2e_contagem_relacoes", None, "", "{:.0f}"),
    ("N-gramas", "Ranking de bigramas: precisão@k contra o léxico de entidades anotadas", "ngramas_ranking_precisao",
     lambda t: t.pivot(index="criterio", columns="k", values="precisao_at_k").reset_index(),
     "Limite inferior: só 15 dos 56 casos têm entidades anotadas.", "{:.2f}"),
    ("N-gramas", "Entidades compostas do gold que contêm uma colocação do corpus", "ngramas_cobertura_gold", None, "",
     "{:.2f}"),
    ("N-gramas", "Pistas lexicais por relação (PMI n-grama × rótulo, 12 textos de treino)", "ngramas_pistas_relacao",
     lambda t: joined(t, "relacao", "pista"), "", "{:.2f}"),
    ("N-gramas", "Arestas do gold que contêm uma pista do próprio rótulo", "ngramas_pistas_cobertura_gold", None, "",
     "{:.2f}"),
    ("N-gramas", "Escolha do limiar de NPMI (12 textos de treino)", "ngramas_candidatos_selecao_limiar", None, "",
     "{:.3f}"),
    ("N-gramas", "Geradores de candidatos a entidade (recall nas entidades do gold)", "ngramas_candidatos", None, "",
     "{:.3f}"),
    ("LR por dentro", "Fração do peso (|coef|) nas features de tipo (LR de relação)", "lr_massa_tipos_texto", None, "",
     "{:.2f}"),
    ("LR por dentro", "Ablação do ngram_range", "lr_ablacao_ngram", None, "", "{:.3f}"),
    ("LR por dentro", "N-gramas que sobrevivem no vocabulário (min_df=2), por ordem", "lr_vocabulario_por_ordem", None,
     "Explica curvas planas na ablação: n-gramas longos raramente aparecem 2 vezes no treino.", "{:.0f}"),
    ("LR por dentro", "N-gramas com maior coeficiente por classe", "lr_top_ngramas",
     lambda t: joined(t, ["tarefa", "classe"], "ngrama", n=6), "", "{:.2f}"),
]

FIGURES_BY_GROUP = {
    "Visão 1 — entidades": ["v1_entidades_f1", "v1_entidades_confusao"],
    "Visão 2 — relação dado o par": ["v2_par_f1", "v2_par_confusao", "v2_falsos_alarmes"],
    "Visão 3 — ponta a ponta": ["v3_e2e_metricas", "v3_e2e_status_por_relacao", "v3_e2e_contagem_relacoes"],
    "N-gramas": ["ngramas_colocacoes", "ngramas_candidatos"],
    "LR por dentro": ["lr_massa_tipos_texto", "lr_ablacao_ngram"],
}


def sections() -> list:
    out = []
    for group, title, name, transform, note, fmt in SPEC:
        t = read(name)
        if t is None:
            continue
        out.append(dict(group=group, title=title, df=transform(t) if transform else t, note=note, fmt=fmt))
    return out


def figures() -> dict:
    return {g: [f for f in names if (FIGURES / f"{f}.png").exists()] for g, names in FIGURES_BY_GROUP.items()}


def build() -> str:
    parts, last_group = [], None
    for s in sections():
        if s["group"] != last_group:
            parts.append(f"### {s['group']}\n")
            last_group = s["group"]
        parts.append(f"#### {s['title']}\n\n{md_table(s['df'], s['fmt'])}\n" + (f"\n{s['note']}\n" if s["note"] else ""))
    if not parts:
        return "_Ainda não há resultados: rode `python3 run.py` a partir de `project2/`._\n"
    return "\n".join(parts)


def warning(failed) -> str:
    if not failed:
        return ""
    items = "\n".join(f"> - **{name}**: `{msg}`" for name, msg in failed)
    return ("> ⚠️ **Resultados incompletos.** As etapas abaixo falharam na última execução; as tabelas e "
            f"figuras correspondentes estão ausentes e NÃO devem ser interpretadas.\n>\n{items}\n\n")


def main(failed=None):
    body = warning(failed) + build()
    out = ROOT / "assets" / "resultados.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Resultados (gerado por src/report.py)\n\n" + body, encoding="utf-8")
    print(f"  relatório -> {out.relative_to(ROOT)}")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        pre, rest = text.split(START, 1)
        _, post = rest.split(END, 1)
        readme.write_text(f"{pre}{START}\n\n{body}\n{END}{post}", encoding="utf-8")
        print("  README.md atualizado entre os marcadores de resultados")
    else:
        print("  aviso: marcadores de resultados não encontrados no README.md")


if __name__ == "__main__":
    main()
