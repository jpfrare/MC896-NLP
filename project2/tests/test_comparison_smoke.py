"""Teste de fumaça do sistema de comparação.  Rode:  cd project2 && pytest tests -q"""
import pandas as pd
import pytest

from src.comparison import config, data, metrics, p1_baseline, run_comparison


def test_dados_gold_e_treino_existem():
    gold = data.load_gold()
    assert len(gold) > 100 and gold["doc_id"].nunique() == 3
    assert set(gold["relation_label"]) >= {config.NO_REL, "SUPPORTS", "HAS_VALUE"}
    manual = pd.concat(data.load_manual_annotations())
    assert not set(manual["doc_id"]) & set(gold["doc_id"]), "vazamento: gold no treino manual"
    weak = data.load_weak_training(gold["doc_id"].unique())
    assert not set(weak["doc_id"]) & set(gold["doc_id"]), "vazamento: gold na weak supervision"


def test_casamento_de_mencoes():
    assert p1_baseline.similarity("abdominal pain", "pain")[0] == 1.0
    assert p1_baseline.similarity("chest pain", "knee surgery")[0] == 0.0
    assert p1_baseline.similarity("", "pain") == (0.0, 0.0)


def test_metricas_perfeitas_e_ruins():
    labels = [config.NO_REL, "A", "B"]
    y = [config.NO_REL, "A", "B", "A"]
    perfect = metrics.summarize(y, y, labels, ["A", "B"])
    assert perfect["macro_f1"] == 1.0 and perfect["detect_f1"] == 1.0
    bad = metrics.summarize(y, [config.NO_REL] * 4, labels, ["A", "B"])
    assert bad["detect_recall"] == 0.0 and "A" in bad["never_predicted"]
    lo, hi = metrics.bootstrap_ci(y, y, labels, n=50)
    assert 0 <= lo <= hi <= 1


def test_p1_ponta_a_ponta_retorna_probabilidades():
    t = p1_baseline.p1_end_to_end(data.load_gold(), data.load_p1_edges())
    assert ((t[["precisao", "recall", "f1"]] >= 0) & (t[["precisao", "recall", "f1"]] <= 1)).all().all()


def test_execucao_completa_sem_mlp(tmp_path):
    out = run_comparison.run(tmp_path / "t", tmp_path / "f", include_mlp=False, include_extras=False,
                             n_boot=20, log=lambda *a, **k: None)
    assert (tmp_path / "t" / "comparison_report.md").exists()
    assert (tmp_path / "t" / "predicoes_relacoes.csv").exists()
    assert len(list((tmp_path / "f").glob("*.png"))) == 8
    assert "Projeto 1 (regras)" in out["summary"].index


def test_mlp_pytorch_treina_e_preve():
    pytest.importorskip("torch")
    from src.comparison.mlp_torch import WeakMLPRelationModel
    weak = data.load_weak_training([])
    m = WeakMLPRelationModel(max_epochs=3, patience=2).fit(weak["intermediate_text"], weak["relation_label"])
    pred = m.predict(weak["intermediate_text"].head(10))
    assert len(pred) == 10 and set(pred) <= set(m.classes_)
