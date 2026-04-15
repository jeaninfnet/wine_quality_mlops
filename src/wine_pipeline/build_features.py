"""
Replica a lógica do Projeto 1 (trabalho_jean):
  - alvo binário opinion (quality > 5)
  - apenas vinho branco
  - remove quality; remove type (constante após filtro)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


TARGET = "opinion"


def build_white_wine_opinion_dataset(
    raw_parquet_path: Path,
    features_parquet_path: Path,
    logger: logging.Logger | None = None,
) -> Path:
    log = logger or logging.getLogger(__name__)
    df = pd.read_parquet(raw_parquet_path)

    if "type" not in df.columns:
        raise ValueError("Coluna 'type' ausente — verifique o dataset.")

    t = df["type"].astype(str).str.lower().str.strip()
    df = df.copy()
    df["type_enc"] = (t == "white").astype(int)

    df = df.dropna()
    df[TARGET] = (df["quality"] > 5).astype(int)

    white = df[df["type_enc"] == 1].copy()
    white = white.drop(columns=["quality", "type", "type_enc"], errors="ignore")

    feature_cols = [c for c in white.columns if c != TARGET]
    white = white[[TARGET] + sorted(feature_cols)]

    features_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    white.to_parquet(features_parquet_path, index=False)
    log.info(
        "Features salvas: %s — %d linhas, %d colunas (target=%s)",
        features_parquet_path,
        len(white),
        white.shape[1],
        TARGET,
    )
    return features_parquet_path
