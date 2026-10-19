"""MLP de relações (PyTorch) — mesma arquitetura/hiperparâmetros do notebook pipeline_rede_neural.

Reproduzido aqui como módulo porque o notebook só salvou o state_dict (.pth): o TfidfVectorizer
e o mapeamento de rótulos NÃO foram salvos, então o modelo salvo não é reutilizável sozinho.
Treino 100% determinístico (seeds fixas, CPU).
"""
import copy
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

from . import config


class WeakMLPRelationModel:
    def __init__(self, seed=config.SEED, max_epochs=300, patience=30, hidden=256, lr=1e-3,
                 weight_power=5.0, ngram_range=(1, 2), max_features=2000, batch_size=32):
        self.seed, self.max_epochs, self.patience = seed, max_epochs, patience
        self.hidden, self.lr, self.weight_power, self.batch_size = hidden, lr, weight_power, batch_size
        self.vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self.net = None
        self.history = []

    @staticmethod
    def _texts(s: pd.Series) -> list[str]:
        return s.replace("", "[ADJACENTE]").fillna("[ADJACENTE]").astype(str).tolist()

    def fit(self, intermediate_text: pd.Series, labels: pd.Series):
        import torch
        import torch.nn as nn

        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        X = torch.tensor(self.vec.fit_transform(self._texts(intermediate_text)).toarray(), dtype=torch.float32)
        y_str = np.asarray(labels)
        self.classes_ = sorted(set(y_str))
        label2id = {l: i for i, l in enumerate(self.classes_)}
        y = torch.tensor([label2id[l] for l in y_str], dtype=torch.long)

        w = compute_class_weight("balanced", classes=np.array(self.classes_), y=y_str) ** self.weight_power
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        net = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(self.hidden, self.hidden // 2), nn.ReLU(),
            nn.Linear(self.hidden // 2, len(self.classes_)),
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)

        best, best_state, wait = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(X), generator=gen)
            total = 0.0
            nb = 0
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = criterion(net(X[idx]), y[idx])
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            ep_loss = total / nb
            self.history.append(ep_loss)
            if ep_loss < best - 1e-4:
                best, best_state, wait = ep_loss, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.best_loss, self.epochs_run = best, len(self.history)
        return self

    def predict(self, intermediate_text: pd.Series) -> pd.Series:
        import torch
        X = torch.tensor(self.vec.transform(self._texts(intermediate_text)).toarray(), dtype=torch.float32)
        with torch.no_grad():
            ids = self.net(X).argmax(dim=1).numpy()
        return pd.Series([self.classes_[i] for i in ids], index=intermediate_text.index)
