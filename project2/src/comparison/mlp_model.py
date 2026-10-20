"""MLP de relações baseado em PyTorch, utilizado a partir do checkpoint salvo."""
import pandas as pd
import torch
import torch.nn as nn
from . import config

class RelationExtractorMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        out = self.dropout(self.relu(self.fc1(x)))
        out = self.relu(self.fc2(out))
        return self.fc3(out)

class PyTorchMLP:
    """Modelo MLP pronto para inferência: texto intermediário -> rótulo de relação."""
    def __init__(self, path=config.MLP_CHECKPOINT):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.vectorizer = ckpt["vectorizer"]
        self.id2label = ckpt["id2label"]
        self.classes_ = sorted(ckpt["label2id"])
        self.net = RelationExtractorMLP(ckpt["input_dim"], ckpt["num_classes"])
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.net.eval()

    def predict(self, intermediate_text: pd.Series) -> pd.Series:
        texts = intermediate_text.replace("", "[ADJACENTE]").fillna("[ADJACENTE]").astype(str).tolist()
        X = torch.tensor(self.vectorizer.transform(texts).toarray(), dtype=torch.float32)
        with torch.no_grad():
            ids = self.net(X).argmax(dim=1).numpy()
        return pd.Series([self.id2label[int(i)] for i in ids], index=intermediate_text.index)