"""Inferência do MLP (weak supervision com as arestas do P1) nos pares do gold.

Lê data/processed/mlp_relation_pipeline_completo.pt (state_dict + vectorizer + rótulos)
e gera pred_MLP_par.csv. Só existe a visão "dado o par": o MLP não tem classe NO_RELATION
nem extrator de pares próprio, então não roda ponta a ponta.

Atenção: o .pt guarda os pesos da ÚLTIMA época do treino. Com --best, usa os pesos de
best_mlp_relation_extractor.pth (menor loss de treino) e salva como MLP_BEST.
"""
import argparse

import pandas as pd
import torch
import torch.nn as nn

from .common import PROCESSED, save_predictions, to_schema
from .gold import load_gold

CHECKPOINT = PROCESSED / "mlp_relation_pipeline_completo.pt"
BEST_WEIGHTS = PROCESSED / "best_mlp_relation_extractor.pth"


class RelationExtractorMLP(nn.Module):
    """Cópia exata da classe da célula 8 de pipelines/pipeline1/pipeline_rede_neural.ipynb."""

    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super(RelationExtractorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def numpy2_pickle_compat():
    """O checkpoint foi salvo com NumPy 2 (o pickle referencia numpy._core.multiarray).
    O ambiente do projeto usa NumPy 1.x (exigido pelo spaCy 3.7), onde o módulo se chama
    numpy.core. Este alias permite ler o pickle sem alterar o checkpoint."""
    import sys
    import numpy
    if int(numpy.__version__.split(".")[0]) < 2 and "numpy._core" not in sys.modules:
        import numpy.core
        import numpy.core.multiarray
        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


def main(best: bool = False):
    numpy2_pickle_compat()
    # weights_only=False: o checkpoint contém um TfidfVectorizer (pickle). Arquivo do próprio repo.
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = RelationExtractorMLP(ck["input_dim"], ck["num_classes"])
    state = torch.load(BEST_WEIGHTS, map_location="cpu") if best else ck["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    pair = load_gold()
    # mesmo preenchimento usado no treino (célula 4 do notebook)
    texts = pair["intermediate_text"].fillna("[ADJACENTE]")
    X_sparse = ck["vectorizer"].transform(texts)
    X = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
    with torch.no_grad():
        ids = model(X).argmax(dim=1).tolist()

    pair["relation_label"] = [ck["id2label"][int(i)] for i in ids]
    zero = int((X_sparse.getnnz(axis=1) == 0).sum())
    print(f"[MLP] classes do checkpoint: {list(ck['label2id'])}; "
          f"{zero}/{len(X)} pares com vetor TF-IDF todo zero (vocabulário fora do treino)")
    name = "MLP_BEST" if best else "MLP"
    save_predictions(to_schema(pair, name, "par"), name, "par")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", action="store_true", help="usar best_mlp_relation_extractor.pth")
    main(ap.parse_args().best)
