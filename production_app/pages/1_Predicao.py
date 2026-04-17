from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ui_styles import aplicar_estilo, cabecalho_pagina
from utils.data_utils import colunas_features, medianas_features
from utils.model_utils import carregar_modelo, prever_proba

st.set_page_config(
    page_title="Predição",
    layout="wide",
    page_icon="📋",
    initial_sidebar_state="expanded",
)

aplicar_estilo()

_URI = f"sqlite:///{(_ROOT / 'mlruns.db').resolve()}".replace("\\", "/")

with st.sidebar:
    st.markdown("##### Parâmetros")
    db_uri = st.text_input("URI do MLflow (SQLite)", value=_URI)
    st.caption("Artefato: `wine-quality-opinion-best` → versão `latest`.")


@st.cache_resource
def _modelo(uri: str):
    return carregar_modelo(uri)


cabecalho_pagina(
    "Consulta de probabilidade",
    "Entrada das variáveis fisico-químicas (vinho branco). Saída: P(opinion = 1), ou seja, estimativa de qualidade percebida > 5.",
)

modelo = _modelo(db_uri)
cols = colunas_features()
med = medianas_features()
for c in cols:
    if f"inp_{c}" not in st.session_state:
        st.session_state[f"inp_{c}"] = float(med.get(c, 0.0))

col_esq, col_dir = st.columns([1.1, 0.9], gap="large")

with col_esq:
    st.markdown("**Formulário de entrada**")
    if st.button("Preencher com medianas do conjunto de features", type="secondary"):
        for c in cols:
            st.session_state[f"inp_{c}"] = float(med.get(c, 0.0))
        st.rerun()

    N_COLS = 3
    for row_start in range(0, len(cols), N_COLS):
        chunk = cols[row_start : row_start + N_COLS]
        row_layout = st.columns(N_COLS)
        for j, c in enumerate(chunk):
            with row_layout[j]:
                st.number_input(c, key=f"inp_{c}", format="%.4f")

with col_dir:
    st.markdown("**Resultado**")
    st.caption("Após ajustar os valores à esquerda, execute a inferência abaixo.")

    if st.button("Executar inferência", type="primary"):
        row_vals = {c: float(st.session_state[f"inp_{c}"]) for c in cols}
        row = pd.DataFrame([row_vals])[cols]
        p1 = prever_proba(row, modelo)
        st.metric(label="Probabilidade estimada P(opinion = 1)", value=f"{p1:.4f}")
        st.progress(min(max(float(p1), 0.0), 1.0), text="Escala 0–1 (limiar usual 0,5)")
        if p1 >= 0.5:
            st.success("No limiar 0,5: classe prevista **1** (qualidade > 5).")
        else:
            st.warning("No limiar 0,5: classe prevista **0** (qualidade ≤ 5).")

st.divider()
st.markdown(
    """
**Nota metodológica:** o pipeline completo (escala, seleção de variáveis, redução opcional e classificador)
foi treinado no notebook de modelagem; esta interface aplica apenas o artefato carregado do MLflow.
    """
)
