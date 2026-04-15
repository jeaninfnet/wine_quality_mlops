"""
Página inicial da aplicação multipágina Streamlit.

Execute na raiz do projeto:
  streamlit run production_app/app.py
"""
import streamlit as st

st.set_page_config(page_title="Wine Quality MLOps", layout="wide", page_icon="🍷")
st.title("🍷 Wine Quality — MLOps")
st.markdown(
    """
    Use o menu lateral para abrir **Predição** ou **Monitoramento**.

    Pré-requisito: executar ingestão, pré-processamento e `notebooks/modelagem.py`
    para gerar `data/features/…` e `mlruns.db`.
    """
)
