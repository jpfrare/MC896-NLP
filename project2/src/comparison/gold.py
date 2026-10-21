"""Carrega o gold standard (data/annotations/gold*.csv) com ids estáveis por linha e por entidade."""
import pandas as pd

from .common import ANNOTATIONS, NO_REL, NOT_ENT


def load_gold(positives_only: bool = False) -> pd.DataFrame:
    frames = []
    for path in sorted(ANNOTATIONS.glob("gold*.csv")):
        df = pd.read_csv(path)
        df["gold_row"] = [f"{path.stem}:{i:03d}" for i in range(len(df))]
        frames.append(df)
    gold = pd.concat(frames, ignore_index=True).rename(columns={"doc_id": "case_id"})
    if positives_only:
        gold = gold[positive_mask(gold)].reset_index(drop=True)
    return gold


def positive_mask(df: pd.DataFrame) -> pd.Series:
    """Linhas que contam como aresta no gold (decisão do time: sem NO_RELATION e sem NOT_ENTITY)."""
    return (df["relation_label"] != NO_REL) & (df["e1_type"] != NOT_ENT) & (df["e2_type"] != NOT_ENT)


def load_gold_entities() -> pd.DataFrame:
    """Spans únicos do gold (e1 e e2), com o tipo anotado. Inclui NOT_ENTITY (filtrado na avaliação).

    Se o mesmo texto aparece com tipos diferentes no mesmo caso, fica o primeiro tipo anotado.
    """
    gold = load_gold()
    ents = pd.concat([
        gold[["case_id", "e1_text", "e1_type"]].set_axis(["case_id", "text", "gold_type"], axis=1),
        gold[["case_id", "e2_text", "e2_type"]].set_axis(["case_id", "text", "gold_type"], axis=1),
    ], ignore_index=True).dropna(subset=["text"])
    ents = ents.drop_duplicates(subset=["case_id", "text"]).reset_index(drop=True)
    ents["ent_id"] = [f"{c}:{i:03d}" for i, c in enumerate(ents["case_id"])]
    return ents


CLINICAL_TYPES = ["SYMPTOM", "FINDING", "DIAGNOSIS", "PROCEDURE"]


def load_clinical_entities(source: str = "gold") -> pd.DataFrame:
    """Definição ÚNICA de "entidade clínica" e "entidade composta" usada pelas análises de n-gramas.

    source = "gold" (3 casos de teste) ou "treino" (12 textos anotados).
    Entidade clínica: tipo em CLINICAL_TYPES (exclui PATIENT, VALUE e NOT_ENTITY).
    Tokens: normalização de src/comparison/match.py (minúsculas, sem artigos nas pontas).
    Composta: 2 ou mais tokens após a normalização.
    """
    from .match import norm_tokens
    if source == "gold":
        ents = load_gold_entities()
    elif source == "treino":
        tr = load_training_annotations()
        ents = pd.concat([tr[["doc_id", "e1_text", "e1_type"]].set_axis(["case_id", "text", "gold_type"], axis=1),
                          tr[["doc_id", "e2_text", "e2_type"]].set_axis(["case_id", "text", "gold_type"], axis=1)],
                         ignore_index=True).dropna(subset=["text"]).drop_duplicates(["case_id", "text"])
    else:
        raise ValueError(source)
    ents = ents[ents["gold_type"].isin(CLINICAL_TYPES)].copy()
    ents["toks"] = ents["text"].map(norm_tokens)
    ents = ents[ents["toks"].map(len) > 0]
    ents["multi"] = ents["toks"].map(len) >= 2
    return ents.reset_index(drop=True)


def load_training_annotations() -> pd.DataFrame:
    """Os 12 textos anotados usados para treinar o LR (text01..12.csv), em ordem fixa."""
    return pd.concat([pd.read_csv(p) for p in sorted(ANNOTATIONS.glob("text*.csv"))], ignore_index=True)


def load_training_annotation_list() -> list:
    """Mesmo conteúdo, como lista de DataFrames (formato esperado por useful.train.CSVReader)."""
    return [pd.read_csv(p) for p in sorted(ANNOTATIONS.glob("text*.csv"))]


def main():
    g = load_gold()
    e = load_gold_entities()
    print(f"gold: {len(g)} linhas, {g.case_id.nunique()} casos; positivas: {positive_mask(g).sum()}")
    print(g.loc[positive_mask(g), "relation_label"].value_counts().to_string())
    print(f"entidades únicas: {len(e)} (sem NOT_ENTITY: {(e.gold_type != NOT_ENT).sum()})")


if __name__ == "__main__":
    main()
