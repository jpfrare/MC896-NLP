"""Constantes, caminhos e esquemas de saída do sistema de comparação.

ESQUEMA DE ARESTAS (pred_<modelo>_<visão>.csv; uma linha = um par classificado ou uma aresta extraída)
    model             P1 | P1_TABELA | LR | LR_TEXTO | MLP | MLP_BEST | TIPOS
    view              par  (o par vem do gold; só o rótulo da relação é predito)
                      e2e  (ponta a ponta: o modelo extrai os próprios pares do texto)
    case_id           id do caso no MultiCaRe
    gold_row          id da linha do gold (só na visão "par"; vazio na "e2e")
    e1_text, e1_type  entidade de origem
    e2_text, e2_type  entidade de destino
    relation_label    rótulo predito (pode ser NO_RELATION)
    intermediate_text texto entre as entidades usado pelo modelo
    negated           True se intermediate_text contém marcador de negação (descritivo)

ESQUEMA DE ENTIDADES (ent_<modelo>.csv; visão "tipagem dado o span")
    model, case_id, ent_id, text, gold_type, pred_type
"""
from pathlib import Path
import re

import pandas as pd

# project2/  (este arquivo está em project2/src/comparison/)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"
OUT = PROCESSED / "comparison"          # todas as saídas tabulares deste sistema
TABLES = OUT / "tables"
FIGURES = ROOT / "assets" / "images"     # figuras usadas no README e nos slides
HTML = ROOT / "assets" / "html"          # painel interativo
PIPELINE1 = ROOT / "pipelines" / "pipeline1"
PIPELINE2 = ROOT / "pipelines" / "pipeline2"

GOLD_IDS = ["PMC3859158_01", "PMC11131390_01", "PMC11722600_01"]

RELATIONS = ["SUPPORTS", "HAS_VALUE", "HAS_SYMPTOM", "UNDERWENT_PROCEDURE", "TREATED_BY"]
ENTITY_TYPES = ["PATIENT", "SYMPTOM", "FINDING", "DIAGNOSIS", "PROCEDURE", "VALUE"]
NO_REL = "NO_RELATION"
NOT_ENT = "NOT_ENTITY"

# Ordem fixa dos modelos em tabelas e gráficos.
MODEL_ORDER = ["GOLD", "TIPOS", "P1", "P1_TABELA", "LR", "LR_TEXTO", "MLP", "MLP_BEST"]

# Uma cor fixa por modelo em TODOS os gráficos (Okabe-Ito; validada para daltonismo).
# Variantes herdam a cor do modelo de origem e se distinguem por hachura (P1_TABELA, LR_TEXTO);
# TIPOS e MLP_BEST também levam hachura (MLP_BEST/rosa tem separação CVD limítrofe com o verde do MLP).
MODEL_COLORS = {
    "GOLD": "#000000",
    "TIPOS": "#999999",
    "P1": "#D55E00",
    "P1_TABELA": "#D55E00",
    "LR": "#0072B2",
    "LR_TEXTO": "#0072B2",
    "MLP": "#009E73",
    "MLP_BEST": "#CC79A7",
}
MODEL_HATCH = {"TIPOS": "//", "P1_TABELA": "..", "LR_TEXTO": "xx", "MLP_BEST": "\\\\"}

# Cores fixas por tipo de entidade (nós dos grafos).
TYPE_COLORS = {
    "PATIENT": "#000000", "SYMPTOM": "#E69F00", "FINDING": "#56B4E9",
    "DIAGNOSIS": "#D55E00", "PROCEDURE": "#009E73", "VALUE": "#CC79A7", NOT_ENT: "#BBBBBB",
}

SCHEMA = ["model", "view", "case_id", "gold_row", "e1_text", "e1_type",
          "e2_text", "e2_type", "relation_label", "intermediate_text", "negated"]
ENT_SCHEMA = ["model", "case_id", "ent_id", "text", "gold_type", "pred_type"]

# Mesmas palavras de negação do P1 (célula 14 de projeto1_modificado.ipynb),
# com as flexões que o P1 cobria via lema (denied -> deny).
NEGATION_RE = re.compile(r"\b(no|not|deny|denies|denied|without|negative|absence)\b", re.IGNORECASE)


def is_negated(text) -> bool:
    return bool(NEGATION_RE.search(str(text))) if pd.notna(text) else False


def ordered(models) -> list:
    """Ordena nomes de modelo pela ordem fixa (desconhecidos vão ao fim)."""
    models = list(dict.fromkeys(models))
    return sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)


def to_schema(df: pd.DataFrame, model: str, view: str) -> pd.DataFrame:
    """Converte qualquer DataFrame com as colunas de aresta para o esquema único."""
    out = df.copy()
    out["model"] = model
    out["view"] = view
    for col in SCHEMA:
        if col not in out.columns:
            out[col] = pd.NA
    out["negated"] = out["intermediate_text"].map(is_negated)
    return out[SCHEMA].reset_index(drop=True)


def save_predictions(df: pd.DataFrame, model: str, view: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"pred_{model}_{view}.csv"
    df.to_csv(path, index=False)
    print(f"[{model}/{view}] {len(df)} linhas -> {path.relative_to(ROOT)}")
    return path


def save_entity_predictions(df: pd.DataFrame, model: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["model"] = model
    path = OUT / f"ent_{model}.csv"
    out[ENT_SCHEMA].to_csv(path, index=False)
    print(f"[{model}/entidades] {len(out)} spans -> {path.relative_to(ROOT)}")
    return path


def load_predictions() -> pd.DataFrame:
    files = sorted(OUT.glob("pred_*.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum pred_*.csv em {OUT}. Rode os adaptadores primeiro.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_entity_predictions() -> pd.DataFrame:
    files = sorted(OUT.glob("ent_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=ENT_SCHEMA)


def save_table(df: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Tabela Markdown sem dependências extras (não exige 'tabulate')."""
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)
