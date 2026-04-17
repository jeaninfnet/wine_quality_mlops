"""Leitura de runs do MLflow para comparação na interface Streamlit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent


def nome_experimento_mlflow() -> str:
    from src.utils.config_loader import load_yaml

    cfg = load_yaml(_PROJECT / "config" / "modeling.yaml")
    return str(cfg["experiment"]["name"])


def carregar_runs_experimento(tracking_uri: str) -> pd.DataFrame:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    name = nome_experimento_mlflow()
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return pd.DataFrame()
    return mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        output_format="pandas",
    )


def _col_metrica(raw: pd.DataFrame, nome: str) -> pd.Series:
    c = f"metrics.{nome}"
    if c in raw.columns:
        return raw[c]
    return pd.Series(np.nan, index=raw.index)


def _col_param(raw: pd.DataFrame, nome: str) -> pd.Series:
    c = f"params.{nome}"
    if c in raw.columns:
        return raw[c]
    return pd.Series(pd.NA, index=raw.index, dtype=object)


def _timestamp_run_ms(raw: pd.DataFrame) -> pd.Series:
    """Para desempate quando há vários runs da mesma configuração."""
    if "end_time" in raw.columns:
        return pd.to_numeric(raw["end_time"], errors="coerce").fillna(0)
    if "start_time" in raw.columns:
        return pd.to_numeric(raw["start_time"], errors="coerce").fillna(0)
    return pd.Series(0, index=raw.index, dtype="int64")


def tabela_comparacao(tracking_uri: str) -> pd.DataFrame:
    """
    Uma linha por combinação (modelo × redução): se existirem vários runs
    (ex.: `modelagem.py` executado várias vezes), mantém-se o **mais recente**.
    Ordenação final por F1 no teste (desc).
    """
    raw = carregar_runs_experimento(tracking_uri)
    if raw.empty:
        return pd.DataFrame()

    run_col = "tags.mlflow.runName" if "tags.mlflow.runName" in raw.columns else None
    if run_col:
        run_labels = raw[run_col]
    elif "run_id" in raw.columns:
        run_labels = raw["run_id"]
    else:
        run_labels = pd.Series(np.arange(len(raw), dtype=str), index=raw.index)

    out = pd.DataFrame(
        {
            "Run": run_labels,
            "Modelo": _col_param(raw, "model"),
            "Redução": _col_param(raw, "reduction"),
            "F1 (teste)": _col_metrica(raw, "f1_test"),
            "Acurácia (teste)": _col_metrica(raw, "accuracy_test"),
            "ROC-AUC (teste)": _col_metrica(raw, "roc_auc_test"),
            "Precisão (teste)": _col_metrica(raw, "precision_test"),
            "Revocação (teste)": _col_metrica(raw, "recall_test"),
            "Treino (s)": _col_metrica(raw, "train_seconds"),
            "Inferência teste (s)": _col_metrica(raw, "infer_seconds_test_set"),
            "_ts_ms": _timestamp_run_ms(raw),
        }
    )
    out = out[out["Modelo"].notna() & (out["Modelo"].astype(str).str.len() > 0)]
    out["Redução"] = out["Redução"].fillna("").astype(str)
    out = out.sort_values("_ts_ms", ascending=False)
    out = out.drop_duplicates(subset=["Modelo", "Redução"], keep="first")
    out = out.drop(columns=["_ts_ms"])
    out = out.sort_values("F1 (teste)", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def dataframe_grafico_f1(tracking_uri: str) -> pd.DataFrame:
    """DataFrame para `st.bar_chart`: índice = rótulo, coluna = F1."""
    t = tabela_comparacao(tracking_uri)
    if t.empty:
        return pd.DataFrame()
    t = t.copy()
    t["Rótulo"] = t["Modelo"].astype(str) + " · " + t["Redução"].astype(str)
    return t.set_index("Rótulo")[["F1 (teste)"]].sort_values("F1 (teste)", ascending=True)
