from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.data_utils import carregar_parquet_features
from utils.model_utils import carregar_modelo

st.title("📡 Monitoramento em lote (simulação)")

_URI = f"sqlite:///{(_ROOT / 'mlruns.db').resolve()}".replace("\\", "/")

with st.sidebar:
    db_uri = st.text_input("MLflow SQLite URI", value=_URI)
    n = st.slider("Amostras", 100, 2000, 400, 50)
    ref_frac = st.slider("Fração referência (primeiros)", 0.3, 0.7, 0.5)

df = carregar_parquet_features()
modelo = carregar_modelo(db_uri)

sub = df.sample(n=min(n, len(df)), random_state=42)
X = sub.drop(columns=["opinion"])
y = sub["opinion"].values
if hasattr(modelo, "predict_proba"):
    proba = modelo.predict_proba(X)[:, 1]
else:
    proba = modelo.predict(X).astype(float)
y_hat = (proba >= 0.5).astype(int)

split = int(len(sub) * ref_frac)
ref_p = proba[:split]
cur_p = proba[split:]
ks_stat, ks_p = ks_2samp(ref_p, cur_p)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("F1", f"{f1_score(y, y_hat, zero_division=0):.3f}")
c2.metric("Accuracy", f"{accuracy_score(y, y_hat):.3f}")
c3.metric("ROC-AUC", f"{roc_auc_score(y, proba):.3f}" if len(np.unique(y)) > 1 else "—")
c4.metric("Precision", f"{precision_score(y, y_hat, zero_division=0):.3f}")
c5.metric("Recall", f"{recall_score(y, y_hat, zero_division=0):.3f}")

st.subheader("Drift de scores (KS entre metades da amostra)")
st.write(
    f"Estatística KS = **{ks_stat:.4f}**, p-valor = **{ks_p:.4g}**. "
    "Valores baixos de p sugerem mudança de distribuição entre lotes simulados."
)
st.page_link("pages/1_Predicao.py", label="Voltar à predição")
