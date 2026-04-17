from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent.parent


def carregar_parquet_features() -> pd.DataFrame:
    from src.utils.config_loader import load_yaml

    pipe = load_yaml(_PROJECT / "config" / "pipeline.yaml")
    p = _PROJECT / pipe["paths"]["features_data_dir"] / pipe["paths"]["features_filename"]
    return pd.read_parquet(p)


def colunas_features() -> list[str]:
    df = carregar_parquet_features()
    return [c for c in df.columns if c != "opinion"]


def medianas_features() -> dict[str, float]:
    """Mediana por coluna (exceto alvo) — útil para preencher formulário com valores típicos."""
    df = carregar_parquet_features()
    feats = df.drop(columns=["opinion"])
    return feats.median(numeric_only=True).astype(float).to_dict()
