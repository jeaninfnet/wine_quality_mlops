"""
Aplicação multipágina — interface tipo dashboard (disciplina / demonstração).

Execute na raiz do projeto:
  streamlit run production_app/app.py
"""
from __future__ import annotations

import streamlit as st

from ui_styles import aplicar_estilo

st.set_page_config(
    page_title="Classificação de qualidade — vinho branco",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

aplicar_estilo()

st.title("Classificação de qualidade percebida (vinho branco)")
st.caption(
    "Projeto MLOps — modelo versionado no MLflow, inferência e monitoramento simulado."
)

st.markdown(
    """
Este painel organiza o **ciclo de uso** do modelo treinado no repositório: consulta unitária
de probabilidade (**Predição**), **comparação dos runs** de modelagem no MLflow e avaliação em lote
com **drift** simulado (**Monitoramento**).

Use o menu **à esquerda** para alternar entre as páginas.
    """
)

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1. Dados**")
    st.caption(
        "Conjunto Kaggle (vinho branco), alvo binário `opinion` (quality > 5). "
        "Requer Parquet de features gerado pelo pré-processamento."
    )
with c2:
    st.markdown("**2. Modelo**")
    st.caption(
        "Carregamento do artefato registrado no MLflow (`wine-quality-opinion-best`). "
        "URI do SQLite configurável na barra lateral de cada página."
    )
with c3:
    st.markdown("**3. Operação simulada**")
    st.caption(
        "Métricas no lote amostrado e teste KS entre subconjuntos de scores, "
        "como proxy de comparação entre “lotes”."
    )

st.divider()

st.markdown("**Navegação**")
p1, p2, p3 = st.columns(3)
with p1:
    st.page_link("pages/1_Predicao.py", label="Predição", icon="📋")
with p2:
    st.page_link("pages/2_Comparacao_modelos.py", label="Comparação de modelos", icon="📊")
with p3:
    st.page_link("pages/3_Monitoramento.py", label="Monitoramento", icon="📈")

st.info(
    "**Pré-requisito:** executar ingestão, pré-processamento e `notebooks/modelagem.py` "
    "para gerar `data/features/…` e `mlruns.db` na raiz do projeto.",
    icon="ℹ️",
)
