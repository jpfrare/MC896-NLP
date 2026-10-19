"""Orquestrador do Pipeline de Comparação."""
import pandas as pd
from . import config, data, p1_baseline, mlp_model, models, metrics, plots, graph_analysis

def run():
    print("[1/5] A criar diretorias e carregar ficheiros...")
    config.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    config.OUT_FIGS.mkdir(parents=True, exist_ok=True)
    
    gold_df, counts = data.load_gold()
    p1_edges = data.load_p1_edges()
    p1_nodes = data.load_p1_nodes()
    
    results = []
    
    print("[2/5] A avaliar Projeto 1 (Regras)...")
    p1_preds = p1_baseline.p1_relation_predictions(gold_df, p1_edges)
    res_p1 = metrics.compute_relation_metrics(gold_df["relation_label"], p1_preds, "Projeto 1 (Regras)")
    results.append(res_p1)
    
    print("[3/5] A avaliar Regressão Logística (N-gramas)...")
    lr_preds = models.train_and_eval_lr(gold_df, use_types=False)
    res_lr = metrics.compute_relation_metrics(gold_df["relation_label"], lr_preds, "LR N-gramas (Manual)")
    results.append(res_lr)
    
    print("[4/5] A avaliar MLP (PyTorch)...")
    mlp = mlp_model.PyTorchMLP()
    mlp_preds = mlp.predict(gold_df["intermediate_text"])
    res_mlp = metrics.compute_relation_metrics(gold_df["relation_label"], mlp_preds, "MLP (PyTorch)")
    results.append(res_mlp)
    
    print("[5/5] A gerar relatório Markdown e guardar gráficos...")
    plots.plot_macro_f1(results, config.OUT_FIGS)
    plots.plot_f1_per_class(results, config.OUT_FIGS)
    plots.plot_confusion_matrices(results, config.OUT_FIGS)
    
    df_metrics = pd.DataFrame([{
        "Modelo": r["model"], 
        "Macro-F1": r["macro_f1"], 
        "Precision": r["macro_precision"], 
        "Recall": r["macro_recall"]
    } for r in results])
    
    graph_stats = graph_analysis.analyze_graph(p1_nodes, p1_edges)
    report_path = config.OUT_TABLES / "relatorio_comparacao.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Relatório de Comparação de Modelos\n\n")
        f.write("A avaliação considera as 5 classes de relações válidas, omitindo NO_RELATION e NOT_ENTITY.\n\n")
        f.write(f"**Distribuição do Gold Analisado:** {counts['linhas_avaliadas']} instâncias.\n\n")
        f.write("## 1. Métricas de Desempenho (Relações)\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n\n## 2. Estatísticas do Grafo (Projeto 1)\n")
        for k, v in graph_stats.items():
            f.write(f"- **{k}**: {v}\n")
            
    print(f"✅ Concluído! Relatório guardado em: {report_path}")

if __name__ == "__main__":
    run()