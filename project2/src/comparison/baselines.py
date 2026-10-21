"""Baseline de referência "só tipos": prevê a relação apenas pelo par (e1_type, e2_type).

Rótulo = classe majoritária daquele par de tipos nos 12 textos de treino do LR
(incluindo NO_RELATION). Par nunca visto -> NO_RELATION.
Serve para medir quanto o texto (n-gramas) acrescenta além dos tipos.
"""
from .common import NO_REL, save_predictions, to_schema
from .gold import load_gold, load_training_annotations


def main():
    train = load_training_annotations()
    lookup = (train.groupby(["e1_type", "e2_type"])["relation_label"]
              .agg(lambda s: s.value_counts().index[0]).to_dict())

    pair = load_gold()
    pair["relation_label"] = [lookup.get((a, b), NO_REL) for a, b in zip(pair["e1_type"], pair["e2_type"])]
    save_predictions(to_schema(pair, "TIPOS", "par"), "TIPOS", "par")


if __name__ == "__main__":
    main()
