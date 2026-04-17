from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

from ui_styles import aplicar_estilo, cabecalho_pagina
from utils.data_utils import carregar_parquet_features
from utils.model_utils import carregar_modelo

st.set_page_config(
    page_title="Monitoramento",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

aplicar_estilo()

_URI = f"sqlite:///{(_ROOT / 'mlruns.db').resolve()}".replace("\\", "/")

with st.sidebar:
    st.markdown("##### Parâmetros da simulação")
    db_uri = st.text_input("URI do MLflow (SQLite)", value=_URI)
    n = st.slider("Tamanho da amostra (lote)", 100, 2000, 400, 50)
    ref_frac = st.slider("Proporção do primeiro sub-lote (referência)", 0.3, 0.7, 0.5)

cabecalho_pagina(
    "Monitoramento em lote (simulação)",
    "Métricas de desempenho no recorte amostrado e comparação de distribuição de scores entre dois sub-lotes (teste KS).",
)

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

st.markdown("**Indicadores no lote atual**")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("F1", f"{f1_score(y, y_hat, zero_division=0):.3f}")
c2.metric("Acurácia", f"{accuracy_score(y, y_hat):.3f}")
c3.metric(
    "ROC-AUC",
    f"{roc_auc_score(y, proba):.3f}" if len(np.unique(y)) > 1 else "—",
)
c4.metric("Precisão", f"{precision_score(y, y_hat, zero_division=0):.3f}")
c5.metric("Revocação", f"{recall_score(y, y_hat, zero_division=0):.3f}")

st.divider()

gc1, gc2 = st.columns([1.0, 1.0])
with gc1:
    st.markdown("**Distribuição de P(opinion = 1)**")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(ref_p, bins=22, alpha=0.65, color="#1a4a7a", label="Sub-lote referência", density=True)
    ax.hist(cur_p, bins=22, alpha=0.55, color="#c45c26", label="Sub-lote comparação", density=True)
    ax.set_xlabel("Probabilidade predita")
    ax.set_ylabel("Densidade")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Sobreposição das distribuições (simulação de lotes)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with gc2:
    st.markdown("**Drift entre sub-lotes (Kolmogorov–Smirnov)**")
    st.write(
        f"Estatística KS = **{ks_stat:.4f}**, p-valor = **{ks_p:.4g}**. "
        "Interpretação: p muito pequeno sugere que as duas metades da amostra ordenada "
        "no tempo de leitura do arquivo se comportam como distribuições diferentes de score — "
        "útil apenas como **exercício didático** de alarme; em produção o desenho seria por lotes reais."
    )
    with st.expander("Como esta simulação se relaciona com drift real"):
        st.markdown(
            """
Em produção, compararíamos scores (ou inputs) entre **janelas temporais** ou **versões de dados**.
Aqui, o split é **determinístico na ordem do sample** após `sample(random_state=42)`,
servindo para demonstrar o fluxo da interface e o uso do teste KS, não para concluir drift de negócio.
            """
        )

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/1_Predicao.py", label="Predição", icon="📋")
with c2:
    st.page_link("pages/2_Comparacao_modelos.py", label="Comparação de modelos", icon="📊")
