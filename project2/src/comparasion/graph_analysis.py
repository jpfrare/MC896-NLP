"""Análise estrutural do grafo gerado (Projeto 1)."""
import pandas as pd

def analyze_graph(p1_nodes: pd.DataFrame, p1_edges: pd.DataFrame) -> dict:
    """Gera estatísticas globais a partir da saída oficial do P1."""
    return {
        "Total de Nós": len(p1_nodes),
        "Total de Arestas": len(p1_edges),
        "Tipos de Nós Únicos": p1_nodes["node_type"].nunique() if "node_type" in p1_nodes.columns else 0,
        "Tipos de Relações Únicas": p1_edges["relation"].nunique() if "relation" in p1_edges.columns else 0
    }