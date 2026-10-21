"""Painel interativo de resultados: assets/html/index.html (um único arquivo, abre no navegador).

Abas:
  Grafos por caso   grafo do gold sobreposto ao de um modelo (P1 ou LR), com filtros por status
                    (certa, rótulo errado, a mais, perdida), por relação, ocultar o nó PATIENT,
                    busca, detalhes ao clicar (com o trecho do texto clínico) e tabela de arestas.
  Grafo do corpus   grafo do P1 nos 55 casos, nós unificados por texto normalizado, filtro por grau.
  Métricas          as mesmas figuras e tabelas do README (src/report.py), num só lugar.

Os dados vão embutidos no HTML (funciona com file://). A biblioteca de grafos (vis-network)
vem de CDN, então o painel precisa de internet na primeira abertura.
"""
import json
from datetime import datetime

import pandas as pd

from src.comparison.common import (GOLD_IDS, HTML, MODEL_COLORS, PROCESSED, RAW, ROOT, TYPE_COLORS,
                                   load_predictions, ordered)
from src.comparison.gold import load_gold
from src.comparison.match import entity_sim, match_edges, norm_tokens
from src import report

STATUS = {
    "GOLD": ("aresta do gold", "#444444"),
    "TP_OK": ("certa", "#1A7F37"),
    "TP_ROTULO": ("rótulo errado", "#C98A00"),
    "FP": ("a mais (não está no gold)", "#C62828"),
    "FN": ("perdida (só no gold)", "#8A8A8A"),
}
VIS_JS = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def node_key(text, etype):
    toks = norm_tokens(text, etype)
    return " ".join(sorted(toks)) or str(text).lower()


def clean(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def case_views(case_id, gold, preds):
    """Arestas por visão (GOLD e cada modelo) com status, e o mapa nó -> rótulo/tipo.
    Entidades preditas que casam com uma do gold (critério de match.py) usam o nó do gold."""
    g = gold[gold["case_id"] == case_id].copy()
    nodes = {}
    for r in g.itertuples():
        for t, ty in ((r.e1_text, r.e1_type), (r.e2_text, r.e2_type)):
            nodes.setdefault(node_key(t, ty), {"label": str(t), "type": ty})
    gold_keys = list(nodes)

    def canon(text, etype):
        toks = norm_tokens(text, etype)
        best, best_s = None, 0.0
        for k in gold_keys:
            s = entity_sim(toks, norm_tokens(nodes[k]["label"], nodes[k]["type"]))
            if s > best_s:
                best, best_s = k, s
        return best or node_key(text, etype)

    def edge(a, b, status, rel, ctx, gold_label=""):
        return {"f": a, "t": b, "s": status, "r": rel, "g": gold_label, "c": clean(ctx)}

    views = {"GOLD": [edge(node_key(r.e1_text, r.e1_type), node_key(r.e2_text, r.e2_type), "GOLD",
                           r.relation_label, r.intermediate_text) for r in g.itertuples()]}
    for model, p in preds.items():
        p = p[p["case_id"] == case_id].reset_index(drop=True)
        edges = []
        for row in match_edges(g, p).itertuples():
            if row.status == "FN":
                gr = g.loc[row.gold_idx]
                edges.append(edge(node_key(gr.e1_text, gr.e1_type), node_key(gr.e2_text, gr.e2_type), "FN",
                                  gr.relation_label, gr.intermediate_text))
            else:
                pr = p.loc[row.pred_idx]
                a, b = canon(pr.e1_text, pr.e1_type), canon(pr.e2_text, pr.e2_type)
                nodes.setdefault(a, {"label": str(pr.e1_text), "type": pr.e1_type})
                nodes.setdefault(b, {"label": str(pr.e2_text), "type": pr.e2_type})
                edges.append(edge(a, b, row.status, pr.relation_label, pr.intermediate_text,
                                  clean(row.gold_label) if row.status == "TP_ROTULO" else ""))
        views[model] = edges
    return nodes, views


def corpus_data():
    edges = pd.read_csv(PROCESSED / "edges.csv")
    nodes, out = {}, {}
    for r in edges.itertuples():
        ends = []
        for norm, ty in ((r.e1_normalizado, r.e1_type), (r.e2_normalizado, r.e2_type)):
            key = "PATIENT:<paciente>" if ty == "PATIENT" else f"{ty}:{norm}"
            nodes.setdefault(key, {"label": "paciente" if ty == "PATIENT" else str(norm), "type": ty, "cases": set()})
            nodes[key]["cases"].add(r.case_id)
            ends.append(key)
        k = (ends[0], ends[1], r.relation_label)
        out[k] = out.get(k, 0) + 1
    deg = {}
    for (a, b, _), n in out.items():
        deg[a] = deg.get(a, 0) + n
        deg[b] = deg.get(b, 0) + n
    return {"n_cases": int(edges["case_id"].nunique()),
            "nodes": [{"id": k, "label": v["label"], "type": v["type"], "deg": deg.get(k, 0),
                       "cases": len(v["cases"])} for k, v in nodes.items()],
            "edges": [{"f": a, "t": b, "r": rel, "n": n} for (a, b, rel), n in out.items()]}


def metrics_data():
    secs = []
    for s in report.sections():
        df = s["df"].copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else s["fmt"].format(v))
        secs.append({"group": s["group"], "title": s["title"], "note": s["note"],
                     "html": df.to_html(index=False, border=0, classes="tbl", escape=True)})
    return {"sections": secs, "figures": report.figures()}


def build(failed=None) -> dict:
    pred = load_predictions()
    e2e = pred[pred["view"] == "e2e"]
    preds = {m: e2e[e2e["model"] == m] for m in ordered(e2e["model"].unique())}
    gold = load_gold(positives_only=True)
    texts = pd.read_csv(RAW / "cases.csv").set_index("case_id")["case_text"].to_dict()
    cases = {}
    for cid in GOLD_IDS:
        nodes, views = case_views(cid, gold, preds)
        cases[cid] = {"text": str(texts.get(cid, "")), "nodes": nodes, "views": views}
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "failed": [{"step": n, "msg": m} for n, m in (failed or [])],
        "models": ["GOLD"] + list(preds),
        "status": {k: {"label": v[0], "color": v[1]} for k, v in STATUS.items()},
        "types": TYPE_COLORS,
        "modelColors": MODEL_COLORS,
        "cases": cases,
        "corpus": corpus_data(),
        "metrics": metrics_data(),
    }


def main(failed=None):
    data = build(failed)
    HTML.mkdir(parents=True, exist_ok=True)
    page = (TEMPLATE.replace("__VIS_JS__", VIS_JS)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    path = HTML / "index.html"
    path.write_text(page, encoding="utf-8")
    print(f"  painel -> {path.relative_to(ROOT)}  (abra no navegador)")


TEMPLATE = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Painel de resultados — MC896 Projeto 2</title>
<script src="__VIS_JS__"></script>
<style>
:root{--bg:#f6f6f4;--panel:#fff;--ink:#1f2328;--muted:#5f6670;--line:#e3e3df;--accent:#0072B2;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
header h1{font-size:16px;margin:0 12px 0 0}
.tabs{display:flex;gap:4px}
.tabs button{border:1px solid var(--line);background:var(--bg);padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.meta{margin-left:auto;color:var(--muted);font-size:12px}
.banner{background:#fff4e5;border-bottom:1px solid #f0c27b;padding:8px 16px;font-size:13px}
main{padding:12px 16px}
.view{display:none}.view.on{display:block}
.grid{display:grid;grid-template-columns:250px minmax(0,1fr) 330px;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.card h3{font-size:13px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.ctl{margin-bottom:14px}
.ctl label{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer}
select,input[type=search]{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff}
.sw{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle}
.sw.dash{border-top-style:dashed}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle}
.cnt{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
#net,#corpusNet{height:72vh;min-height:420px;background:#fff;border:1px solid var(--line);border-radius:8px}
#details{max-height:72vh;overflow:auto}
#details .ctx{background:var(--bg);border-radius:6px;padding:8px;margin:6px 0;font-size:13px}
mark{background:#ffe08a;padding:0 1px}
.tbl{border-collapse:collapse;width:100%;font-size:13px}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
.tbl th{background:var(--bg);position:sticky;top:0}
.tbl tr.click{cursor:pointer}.tbl tr.click:hover{background:#f0f6fb}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;color:#fff;font-size:12px;white-space:nowrap}
#edgeTable{max-height:40vh;overflow:auto;margin-top:12px}
.hint{color:var(--muted);font-size:12px}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;margin:8px 0 16px}
.figs img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:zoom-in}
.mgroup h2{font-size:17px;margin:22px 0 6px}
.mtable{overflow:auto;margin-bottom:14px}
@media (max-width:1100px){.grid{grid-template-columns:1fr}#net,#corpusNet{height:60vh}}
</style>
</head>
<body>
<header>
  <h1>MC896 · Projeto 2 — resultados</h1>
  <nav class="tabs">
    <button data-tab="casos" class="on">Grafos por caso</button>
    <button data-tab="corpus">Grafo do corpus (P1)</button>
    <button data-tab="metricas">Métricas</button>
  </nav>
  <span class="meta" id="meta"></span>
</header>
<div id="banner" class="banner" hidden></div>
<main>
<section id="casos" class="view on">
  <div class="grid">
    <div class="card">
      <div class="ctl"><h3>Caso</h3><select id="caseSel"></select></div>
      <div class="ctl"><h3>Comparar gold com</h3><select id="modelSel"></select>
        <div class="hint">"GOLD" mostra só a anotação manual.</div></div>
      <div class="ctl"><h3>Arestas</h3><div id="statusBox"></div></div>
      <div class="ctl"><h3>Relações</h3><div id="relBox"></div></div>
      <div class="ctl"><h3>Exibição</h3>
        <label><input type="checkbox" id="hidePatient"> ocultar o nó PATIENT</label>
        <label><input type="checkbox" id="edgeLabels"> mostrar rótulos das arestas</label>
      </div>
      <div class="ctl"><h3>Buscar entidade</h3><input type="search" id="search" placeholder="ex.: pancreatitis"></div>
      <div class="ctl"><h3>Tipos de entidade</h3><div id="typeLegend"></div></div>
    </div>
    <div>
      <div id="net"></div>
      <div class="card" id="edgeTable"></div>
    </div>
    <div class="card" id="details"><h3>Detalhes</h3><p class="hint">Clique num nó ou numa aresta (ou numa linha da tabela) para ver o contexto e o trecho do texto clínico.</p></div>
  </div>
</section>
<section id="corpus" class="view">
  <div class="grid" style="grid-template-columns:250px minmax(0,1fr)">
    <div class="card">
      <div class="ctl"><h3>Grau mínimo</h3><input type="range" id="minDeg" min="1" max="12" value="3"> <span id="minDegVal">3</span>
        <div class="hint">Mostra só entidades com pelo menos esse número de arestas no corpus.</div></div>
      <div class="ctl"><h3>Relações</h3><div id="corpusRelBox"></div></div>
      <div class="ctl" id="corpusInfo"></div>
      <div class="ctl"><h3>Tipos de entidade</h3><div id="typeLegend2"></div></div>
    </div>
    <div id="corpusNet"></div>
  </div>
</section>
<section id="metricas" class="view"><div id="metricsBody"></div></section>
</main>
<script>
const DATA = __DATA__;
const $$ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const short = (s, n=26) => s.length > n ? s.slice(0, n-1) + "…" : s;
const ST = DATA.status, TY = DATA.types;
const RELS = ["SUPPORTS","HAS_VALUE","HAS_SYMPTOM","UNDERWENT_PROCEDURE","TREATED_BY"];

$$("#meta").textContent = "gerado em " + DATA.generated;
if (DATA.failed.length) {
  const b = $$("#banner"); b.hidden = false;
  b.innerHTML = "<b>Resultados incompletos.</b> Etapas que falharam: " + DATA.failed.map(f => esc(f.step) + " (" + esc(f.msg) + ")").join("; ");
}
if (typeof vis === "undefined") {
  $$("#net").innerHTML = $$("#corpusNet").innerHTML = '<p style="padding:16px">Não foi possível carregar a biblioteca de grafos (vis-network). Verifique a conexão com a internet e recarregue.</p>';
}
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("on", v.id === b.dataset.tab));
  if (b.dataset.tab === "corpus") buildCorpus();
  if (b.dataset.tab === "casos" && net) net.redraw();
});
const legendTypes = el => el.innerHTML = Object.entries(TY).filter(([t]) => t !== "NOT_ENTITY")
  .map(([t, c]) => `<label><span class="dot" style="background:${c}"></span>${t}</label>`).join("");
legendTypes($$("#typeLegend")); legendTypes($$("#typeLegend2"));

// ---------------- grafos por caso ----------------
let net = null, nodesDS = null, edgesDS = null, shown = [];
const caseSel = $$("#caseSel"), modelSel = $$("#modelSel");
Object.keys(DATA.cases).forEach(c => caseSel.add(new Option(c, c)));
DATA.models.forEach(m => modelSel.add(new Option(m, m)));
modelSel.value = DATA.models.includes("LR") ? "LR" : DATA.models[DATA.models.length - 1];

$$("#statusBox").innerHTML = Object.entries(ST).map(([k, v]) =>
  `<label data-st="${k}"><input type="checkbox" class="stChk" value="${k}" checked>
   <span class="sw ${k === "FN" ? "dash" : ""}" style="border-color:${v.color}"></span>${v.label}<span class="cnt" id="cnt_${k}"></span></label>`).join("");
$$("#relBox").innerHTML = RELS.map(r => `<label><input type="checkbox" class="relChk" value="${r}" checked>${r}</label>`).join("");

function nodeStyle(n) {
  return {color: {background: TY[n.type] || "#bbb", border: "#ffffff", highlight: {background: TY[n.type] || "#bbb", border: "#111"}},
          borderWidth: 2, shape: "dot", size: n.type === "PATIENT" ? 18 : 11, font: {size: 12, color: "#222"}};
}
function edgeTitle(e) {
  const d = document.createElement("div");
  d.style.maxWidth = "360px";
  d.innerHTML = `<b>${esc(e.r)}</b> — ${esc(ST[e.s].label)}` + (e.g ? `<br>gold: ${esc(e.g)}` : "") +
                (e.c ? `<br><i>contexto:</i> ${esc(e.c)}` : "");
  return d;
}
function buildCase() {
  if (typeof vis === "undefined") return;
  const C = DATA.cases[caseSel.value];
  nodesDS = new vis.DataSet(Object.entries(C.nodes).map(([id, n]) =>
    Object.assign({id, label: short(n.label), title: `${n.label} [${n.type}]`}, nodeStyle(n))));
  const union = []; const seen = new Set();
  Object.values(C.views).flat().forEach(e => { const k = e.f + "→" + e.t; if (!seen.has(k)) { seen.add(k); union.push({from: e.f, to: e.t}); } });
  edgesDS = new vis.DataSet(union.map((e, i) => Object.assign({id: "u" + i}, e)));
  if (net) net.destroy();
  net = new vis.Network($$("#net"), {nodes: nodesDS, edges: edgesDS}, {
    physics: {solver: "forceAtlas2Based", forceAtlas2Based: {gravitationalConstant: -70, springLength: 120, avoidOverlap: 0.7},
              stabilization: {iterations: 700, fit: true}},
    interaction: {hover: true, tooltipDelay: 120, navigationButtons: true, keyboard: true},
    edges: {smooth: {type: "continuous"}, arrows: {to: {enabled: true, scaleFactor: 0.55}}, width: 1.8},
  });
  net.once("stabilizationIterationsDone", () => { net.setOptions({physics: false}); render(true); });
  net.on("click", p => p.nodes.length ? showNode(p.nodes[0]) : p.edges.length ? showEdge(p.edges[0]) : null);
}
function filters() {
  return {st: new Set([...document.querySelectorAll(".stChk:checked")].map(x => x.value)),
          rel: new Set([...document.querySelectorAll(".relChk:checked")].map(x => x.value)),
          hideP: $$("#hidePatient").checked, labels: $$("#edgeLabels").checked};
}
function render(fit) {
  if (!net) return;
  const C = DATA.cases[caseSel.value], f = filters();
  const all = C.views[modelSel.value] || [];
  const isP = id => C.nodes[id] && C.nodes[id].type === "PATIENT";
  const pre = all.filter(e => f.rel.has(e.r) || (e.g && f.rel.has(e.g))).filter(e => !(f.hideP && (isP(e.f) || isP(e.t))));
  Object.keys(ST).forEach(k => {
    const n = pre.filter(e => e.s === k).length, lab = document.querySelector(`[data-st="${k}"]`);
    $$("#cnt_" + k).textContent = n; lab.style.display = n ? "flex" : "none";
  });
  shown = pre.filter(e => f.st.has(e.s));
  edgesDS.clear();
  edgesDS.add(shown.map((e, i) => ({id: "e" + i, from: e.f, to: e.t, title: edgeTitle(e),
    color: {color: ST[e.s].color, highlight: ST[e.s].color, hover: ST[e.s].color},
    dashes: e.s === "FN", width: e.s === "FP" ? 2.4 : 1.8, label: f.labels ? e.r : undefined,
    font: {size: 9, align: "middle", color: "#333", strokeWidth: 3, strokeColor: "#fff"}})));
  const used = new Set(shown.flatMap(e => [e.f, e.t]));
  nodesDS.update(nodesDS.getIds().map(id => ({id, hidden: !used.has(id)})));
  if (fit) net.fit({animation: false});
  table();
}
function pill(s) { return `<span class="pill" style="background:${ST[s].color}">${esc(ST[s].label)}</span>`; }
function table() {
  const C = DATA.cases[caseSel.value];
  const rows = shown.map((e, i) => `<tr class="click" data-i="${i}"><td>${pill(e.s)}</td><td>${esc(C.nodes[e.f].label)}</td>
     <td>${esc(e.r)}${e.g ? `<br><span class="hint">gold: ${esc(e.g)}</span>` : ""}</td><td>${esc(C.nodes[e.t].label)}</td><td>${esc(e.c)}</td></tr>`).join("");
  $$("#edgeTable").innerHTML = `<h3>${shown.length} arestas visíveis</h3><table class="tbl"><thead><tr><th>status</th><th>origem</th><th>relação</th><th>destino</th><th>contexto</th></tr></thead><tbody>${rows}</tbody></table>`;
  document.querySelectorAll("#edgeTable tr.click").forEach(tr => tr.onclick = () => {
    const id = "e" + tr.dataset.i; net.selectEdges([id]); showEdge(id);
    const e = shown[+tr.dataset.i]; net.focus(e.f, {scale: 1.1, animation: true});
  });
}
function excerpts(text, terms) {
  const sents = text.split(/(?<=[.!?])\s+/);
  const res = terms.filter(t => t && t.length > 1).map(t => new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"));
  const hits = sents.filter(s => res.some(r => { r.lastIndex = 0; return r.test(s); })).slice(0, 4);
  if (!hits.length) return '<p class="hint">Trecho não encontrado literalmente no texto.</p>';
  return hits.map(s => { let h = esc(s); res.forEach(r => { h = h.replace(new RegExp(r.source, "gi"), m => `<mark>${m}</mark>`); }); return `<div class="ctx">${h}</div>`; }).join("");
}
function showNode(id) {
  const C = DATA.cases[caseSel.value], n = C.nodes[id];
  const es = shown.filter(e => e.f === id || e.t === id);
  $$("#details").innerHTML = `<h3>Entidade</h3><p><span class="dot" style="background:${TY[n.type]}"></span> <b>${esc(n.label)}</b> · ${esc(n.type)}</p>
    <h3>${es.length} arestas visíveis</h3>` + es.map(e => `<div class="ctx">${pill(e.s)} ${esc(C.nodes[e.f].label)} <b>—${esc(e.r)}→</b> ${esc(C.nodes[e.t].label)}</div>`).join("") +
    `<h3>No texto clínico</h3>` + (n.type === "PATIENT" ? '<p class="hint">Nó único do paciente do caso.</p>' : excerpts(C.text, [n.label]));
}
function showEdge(eid) {
  const C = DATA.cases[caseSel.value], e = shown[+String(eid).slice(1)];
  if (!e) return;
  const a = C.nodes[e.f], b = C.nodes[e.t];
  $$("#details").innerHTML = `<h3>Aresta</h3><p>${pill(e.s)}</p><p><b>${esc(a.label)}</b> <span class="hint">[${a.type}]</span><br>—${esc(e.r)}→<br><b>${esc(b.label)}</b> <span class="hint">[${b.type}]</span></p>` +
    (e.g ? `<p>Rótulo no gold: <b>${esc(e.g)}</b></p>` : "") + (e.c ? `<h3>Contexto usado</h3><div class="ctx">${esc(e.c)}</div>` : "") +
    `<h3>No texto clínico</h3>` + excerpts(C.text, [a.type === "PATIENT" ? "" : a.label, b.type === "PATIENT" ? "" : b.label]);
}
$$("#search").oninput = ev => {
  const q = ev.target.value.trim().toLowerCase(); if (!net || q.length < 2) return;
  const C = DATA.cases[caseSel.value];
  const ids = nodesDS.get({filter: n => !n.hidden && C.nodes[n.id].label.toLowerCase().includes(q)}).map(n => n.id);
  net.selectNodes(ids); if (ids.length) { net.focus(ids[0], {scale: 1.2, animation: true}); showNode(ids[0]); }
};
caseSel.onchange = buildCase;
modelSel.onchange = () => render(false);
document.querySelectorAll(".stChk,.relChk,#hidePatient,#edgeLabels").forEach(x => x.onchange = () => render(false));
buildCase();

// ---------------- grafo do corpus ----------------
let cnet = null;
$$("#corpusRelBox").innerHTML = RELS.map(r => `<label><input type="checkbox" class="crelChk" value="${r}" checked>${r}</label>`).join("");
function buildCorpus() {
  if (typeof vis === "undefined") return;
  const k = +$$("#minDeg").value; $$("#minDegVal").textContent = k;
  const rel = new Set([...document.querySelectorAll(".crelChk:checked")].map(x => x.value));
  const keep = new Set(DATA.corpus.nodes.filter(n => n.deg >= k).map(n => n.id));
  const es = DATA.corpus.edges.filter(e => rel.has(e.r) && keep.has(e.f) && keep.has(e.t));
  const used = new Set(es.flatMap(e => [e.f, e.t]));
  const ns = DATA.corpus.nodes.filter(n => used.has(n.id)).map(n => Object.assign(
    {id: n.id, label: short(n.label, 22), title: `${n.label} [${n.type}] · grau ${n.deg} · ${n.cases} casos`, value: n.deg}, nodeStyle(n)));
  $$("#corpusInfo").innerHTML = `<h3>Exibindo</h3>${ns.length} entidades e ${es.length} arestas (${DATA.corpus.n_cases} casos). Espessura = nº de ocorrências.`;
  const ed = es.map((e, i) => ({id: i, from: e.f, to: e.t, value: e.n, title: `${e.r} ×${e.n}`, color: {color: "#9aa0a6", opacity: 0.7}}));
  if (cnet) cnet.destroy();
  cnet = new vis.Network($$("#corpusNet"), {nodes: new vis.DataSet(ns), edges: new vis.DataSet(ed)}, {
    physics: {solver: "forceAtlas2Based", forceAtlas2Based: {gravitationalConstant: -50, springLength: 90}, stabilization: {iterations: 500}},
    nodes: {scaling: {min: 6, max: 34}}, edges: {scaling: {min: 1, max: 8}, arrows: {to: {enabled: true, scaleFactor: 0.4}}},
    interaction: {hover: true, navigationButtons: true}});
  cnet.once("stabilizationIterationsDone", () => cnet.setOptions({physics: false}));
}
$$("#minDeg").oninput = buildCorpus;
document.querySelectorAll(".crelChk").forEach(x => x.onchange = buildCorpus);

// ---------------- métricas ----------------
(function () {
  const M = DATA.metrics, groups = [...new Set([...Object.keys(M.figures), ...M.sections.map(s => s.group)])];
  $$("#metricsBody").innerHTML = groups.map(g => `<div class="mgroup"><h2>${esc(g)}</h2>
    <div class="figs">${(M.figures[g] || []).map(f => `<a href="../images/${f}.png" target="_blank"><img src="../images/${f}.png" alt="${f}" loading="lazy"></a>`).join("")}</div>
    ${M.sections.filter(s => s.group === g).map(s => `<div class="card mtable"><h3>${esc(s.title)}</h3>${s.html}${s.note ? `<p class="hint">${esc(s.note)}</p>` : ""}</div>`).join("")}
  </div>`).join("") || '<p>Sem resultados ainda: rode <code>python3 run.py</code>.</p>';
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
