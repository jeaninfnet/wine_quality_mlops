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

from utils.data_utils import colunas_features
from utils.model_utils import carregar_modelo, prever_proba

st.title("🍷 Classificação: opinião de qualidade (vinho branco)")
st.caption("Probabilidade estimada de quality > 5 (rótulo `opinion=1`).")

_URI = f"sqlite:///{(_ROOT / 'mlruns.db').resolve()}".replace("\\", "/")

with st.sidebar:
    db_uri = st.text_input("MLflow SQLite URI", value=_URI)


@st.cache_resource
def _modelo(uri: str):
    return carregar_modelo(uri)


cols = colunas_features()
modelo = _modelo(db_uri)

st.subheader("Entrada das features")
vals = {}
# Uma linha de colunas por vez — não chamar st.columns dentro do loop por campo
# (isso recria o layout a cada iteração e desalinha os widgets).
N_COLS = 3
for row_start in range(0, len(cols), N_COLS):
    chunk = cols[row_start : row_start + N_COLS]
    row_layout = st.columns(N_COLS)
    for j, c in enumerate(chunk):
        with row_layout[j]:
            vals[c] = st.number_input(c, value=0.0, format="%.4f")

if st.button("Calcular probabilidade"):
    row = pd.DataFrame([vals])[cols]
    p1 = prever_proba(row, modelo)
    st.metric("P(opinion = 1)", f"{p1:.3f}")
    st.progress(min(max(p1, 0.0), 1.0))
