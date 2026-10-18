"""Baselines do Projeto 2 e confronto com as regras do Projeto 1.

Rodar a partir de project2/:   python src/baselines.py

Baselines:
  majority      sempre a classe mais frequente no treino
  types_only    olha so (tipo_e1, tipo_e2)  -> mostra quanto os tipos sozinhos explicam
  tfidf_lr_text regressao logistica sobre TF-IDF (1-2 gramas) do texto entre as entidades
  tfidf_lr_+types idem, mais os tipos como tokens
  rules_p1      as arestas do Projeto 1 (data/processed/edges.csv) casadas com o gold
"""
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation import (compare_table, evaluate, leave_one_doc_out, print_report,
                        summary_row, train_test)
from gold import ANNOTATIONS_TEST, LABELS, NODE_TYPES, PROJECT, build_entities, load_gold

EDGES_PATH = PROJECT / "data" / "processed" / "edges.csv"
RESULTS_PATH = PROJECT / "data" / "processed" / "baseline_results.csv"

def majority_fit_predict(train, test):
    label = Counter(train["relation"]).most_common(1)[0][0]
    return [label] * len(test)


def types_only_fit_predict(train, test):
    fallback = Counter(train["relation"]).most_common(1)[0][0]
    by_pair = (
        train.groupby(["e1_type", "e2_type"])["relation"]
        .agg(lambda s: Counter(s).most_common(1)[0][0])
        .to_dict()
    )
    return [by_pair.get((a, b), fallback) for a, b in zip(test["e1_type"], test["e2_type"])]


def _text(df, with_types):
    txt = df["context"].str.lower()
    if with_types:
        txt = txt + " T1_" + df["e1_type"] + " T2_" + df["e2_type"]
    return txt


def make_tfidf_lr(with_types):
    """Exemplo de modelo plugavel no harness (seu modelo deve ter esta mesma forma)."""
    def fit_predict(train, test):
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False)
        X_train = vec.fit_transform(_text(train, with_types))
        X_test = vec.transform(_text(test, with_types))
        clf = LogisticRegression(max_iter=1000, C=10)
        clf.fit(X_train, train["relation"])
        return list(clf.predict(X_test))
    return fit_predict

def _tokens(s):
    toks = re.findall(r"[a-z0-9]+", str(s).lower())
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks}


def _match(a, b):
    """Casamento aproximado: o P1 guarda lemas em minusculas, o gold o texto original.
    Casa se um conjunto de tokens contem o outro, ou se Jaccard >= 0.5."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta or len(ta & tb) / len(ta | tb) >= 0.5


def match_rules_to_gold(gold, edges):
    """Para cada par do gold devolve (relacao da regra ou NO_RELATION, indices das arestas usadas)."""
    preds, used = [], set()
    for _, g in gold.iterrows():
        cand = edges[edges["case_id"] == g["doc_id"]]
        hits = [
            i for i, e in cand.iterrows()
            if (_match(g["e1_text"], e["source_label"]) and _match(g["e2_text"], e["target_label"]))
            or (_match(g["e1_text"], e["target_label"]) and _match(g["e2_text"], e["source_label"]))
        ]
        used.update(hits)
        preds.append(Counter(edges.loc[hits, "relation"]).most_common(1)[0][0] if hits else "NO_RELATION")
    return preds, used


def rules_report(gold, edges):
    """Avalia as regras do P1 sobre o gold. Nao ha treino, entao nao ha folds."""
    preds, used = match_rules_to_gold(gold, edges)
    res = evaluate(gold["relation"], preds)
    docs = edges[edges["case_id"].isin(gold["doc_id"].unique())]
    positives = gold["relation"] != "NO_RELATION"
    found = pd.Series(preds, index=gold.index) != "NO_RELATION"
    res["rules_extra"] = {
        "arestas_das_regras_nos_textos_do_gold": len(docs),
        "casadas_com_algum_par_do_gold": len(used),
        "sem_par_no_gold (nao julgaveis)": len(docs) - len(used),
        "pares_positivos_do_gold_encontrados": f"{int((found & positives).sum())}/{int(positives.sum())}",
    }
    res["preds"] = preds
    return res


def graph_comparison(gold, edges, rules_preds):
    """Contagem de arestas por relacao: gold vs regras (nos textos anotados) vs regras (todos os textos)."""
    docs = edges[edges["case_id"].isin(gold["doc_id"].unique())]
    table = pd.DataFrame({
        "gold": gold["relation"].value_counts(),
        "regras_casadas_com_gold": pd.Series(rules_preds).value_counts(),
        "regras_todas_arestas_nos_mesmos_textos": docs["relation"].value_counts(),
        "regras_todos_os_56_textos": edges["relation"].value_counts(),
    }).reindex(LABELS).fillna(0).astype(int)
    return table

def node_majority(train, test):
    return [Counter(train["type"]).most_common(1)[0][0]] * len(test)


def node_tfidf_lr(train, test):
    """Tipo da entidade a partir so do proprio texto (1-3 gramas), sem spaCy."""
    vec = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(train["text"]), train["type"])
    return list(clf.predict(vec.transform(test["text"])))


def run_node_baselines(gold):
    ents = build_entities(gold)
    print(f"\n##### ETAPA 1 (tipo do no): {len(ents)} entidades unicas #####")
    for name, fn in [("node_majority", node_majority), ("node_tfidf_lr", node_tfidf_lr)]:
        res = leave_one_doc_out(ents, fn, labels=NODE_TYPES, target="type")
        print_report(res, name)

def main():
    gold = load_gold()
    edges = pd.read_csv(EDGES_PATH)
    print(f"Gold: {len(gold)} pares em {gold.doc_id.nunique()} textos (validacao: um texto fora por vez)")

    rows = []
    for name, fn in [
        ("majority", majority_fit_predict),
        ("types_only", types_only_fit_predict),
        ("tfidf_lr_text", make_tfidf_lr(False)),
        ("tfidf_lr_+types", make_tfidf_lr(True)),
    ]:
        res = leave_one_doc_out(gold, fn)
        print_report(res, name)
        rows.append(summary_row(res, name))

    run_node_baselines(gold)

    rres = rules_report(gold, edges)
    print_report(rres, "rules_p1 (regras do Projeto 1)")
    print("\nDetalhe das regras:")
    for k, v in rres["rules_extra"].items():
        print(f"  {k}: {v}")
    rows.append(summary_row(rres, "rules_p1"))

    print("\nArestas por relacao:")
    print(graph_comparison(gold, edges, rres["preds"]).to_string())

    if ANNOTATIONS_TEST.exists():
        test = load_gold("test")
        print(f"\n##### TESTE SEPARADO: {len(test)} pares em {test.doc_id.nunique()} textos #####")
        results = {}
        for name, fn in [
            ("majority", majority_fit_predict),
            ("types_only", types_only_fit_predict),
            ("tfidf_lr_text", make_tfidf_lr(False)),
            ("tfidf_lr_+types", make_tfidf_lr(True)),
        ]:
            results[name] = train_test(gold, test, fn)
        results["rules_p1"] = rules_report(test, edges)
        table = compare_table(results)
        print(table.to_string(index=False))
        table.to_csv(PROJECT / "data" / "processed" / "comparison_test.csv", index=False)
    else:
        print("\n(sem data/annotations_test/: a comparacao principal precisa do teste separado)")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_PATH, index=False)
    print(f"\nTabela salva em {RESULTS_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()