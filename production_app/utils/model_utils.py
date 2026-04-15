from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

_REGISTRY = "wine-quality-opinion-best"


def carregar_modelo(db_uri: str) -> Any:
    mlflow.set_tracking_uri(db_uri)
    return mlflow.sklearn.load_model(f"models:/{_REGISTRY}/latest")


def prever_proba(features_df: pd.DataFrame, modelo: Any) -> float:
    if hasattr(modelo, "predict_proba"):
        p = modelo.predict_proba(features_df)
        return float(np.asarray(p)[0, 1])
    out = modelo.predict(features_df)
    return float(np.asarray(out).ravel()[0])
