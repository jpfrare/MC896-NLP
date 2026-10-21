"""Casamento de arestas preditas com o gold na visão ponta a ponta.

Critério (documentado no README):
  - normalização: minúsculas, tokens alfanuméricos, sem artigos/possessivos nas pontas;
    toda entidade do tipo PATIENT vira o nó único "<patient>" (há um paciente por caso);
  - duas entidades casam se Jaccard(tokens) >= 0,5 ou se uma contém a outra;
  - um par casa com outro na mesma direção OU invertido (direção é avaliada à parte);
  - casamento um-para-um, guloso pela maior similaridade média, dentro do mesmo caso.
"""
import re

import pandas as pd

EDGE_WORDS = {"a", "an", "the", "any", "his", "her", "their", "its", "this", "that",
              "these", "those", "some", "of", "with", "and", "or"}
JACCARD_MIN = 0.5


def norm_tokens(text, etype=None) -> frozenset:
    if etype == "PATIENT":
        return frozenset({"<patient>"})
    toks = re.findall(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*", str(text).lower())
    while toks and toks[0] in EDGE_WORDS:
        toks.pop(0)
    while toks and toks[-1] in EDGE_WORDS:
        toks.pop()
    return frozenset(toks)


def entity_sim(a: frozenset, b: frozenset) -> float:
    """Similaridade em [0,1]; 0 se não passa no critério de casamento."""
    if not a or not b:
        return 0.0
    jac = len(a & b) / len(a | b)
    if jac >= JACCARD_MIN or a <= b or b <= a:
        return max(jac, 0.5)
    return 0.0


def match_edges(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma linha por aresta do gold e por aresta predita, com o status:

    TP_OK     casou e o rótulo da relação está certo
    TP_ROTULO casou mas o rótulo está errado
    FN        aresta do gold sem correspondente
    FP        aresta predita sem correspondente no gold
    """
    rows = []
    for case_id in sorted(set(gold["case_id"]) | set(pred["case_id"])):
        g = gold[gold["case_id"] == case_id]
        p = pred[pred["case_id"] == case_id]
        gk = [(i, norm_tokens(r.e1_text, r.e1_type), norm_tokens(r.e2_text, r.e2_type)) for i, r in g.iterrows()]
        pk = [(j, norm_tokens(r.e1_text, r.e1_type), norm_tokens(r.e2_text, r.e2_type)) for j, r in p.iterrows()]
        cands = []
        for i, g1, g2 in gk:
            for j, p1, p2 in pk:
                direct = (entity_sim(g1, p1), entity_sim(g2, p2))
                rev = (entity_sim(g1, p2), entity_sim(g2, p1))
                for (s1, s2), reversed_ in ((direct, False), (rev, True)):
                    if s1 > 0 and s2 > 0:
                        cands.append(((s1 + s2) / 2, reversed_, i, j))
        cands.sort(key=lambda c: (-c[0], c[1]))
        used_g, used_p = set(), set()
        for score, reversed_, i, j in cands:
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            gr, pr = gold.loc[i], pred.loc[j]
            ok_rel = gr.relation_label == pr.relation_label
            pe1, pe2 = (pr.e2_type, pr.e1_type) if reversed_ else (pr.e1_type, pr.e2_type)
            rows.append(dict(case_id=case_id, gold_idx=i, pred_idx=j, score=score, reversed=reversed_,
                             status="TP_OK" if ok_rel else "TP_ROTULO",
                             types_ok=(gr.e1_type == pe1) and (gr.e2_type == pe2),
                             gold_label=gr.relation_label, pred_label=pr.relation_label))
        for i, *_ in gk:
            if i not in used_g:
                rows.append(dict(case_id=case_id, gold_idx=i, pred_idx=pd.NA, score=0.0, reversed=False,
                                 status="FN", types_ok=False, gold_label=gold.loc[i, "relation_label"], pred_label=pd.NA))
        for j, *_ in pk:
            if j not in used_p:
                rows.append(dict(case_id=case_id, gold_idx=pd.NA, pred_idx=j, score=0.0, reversed=False,
                                 status="FP", types_ok=False, gold_label=pd.NA, pred_label=pred.loc[j, "relation_label"]))
    return pd.DataFrame(rows)
