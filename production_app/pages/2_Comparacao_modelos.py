from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_ROOT = _APP_DIR.parent
for _p in (str(_APP_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ui_styles import aplicar_estilo, cabecalho_pagina
from utils.mlflow_runs import dataframe_grafico_f1, nome_experimento_mlflow, tabela_comparacao

st.set_page_config(
    page_title="Comparação de modelos",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

aplicar_estilo()

_URI = f"sqlite:///{(_ROOT / 'mlruns.db').resolve()}".replace("\\", "/")

with st.sidebar:
    st.markdown("##### Conexão")
    db_uri = st.text_input("URI do MLflow (SQLite)", value=_URI)
    st.caption(f"Experimento: `{nome_experimento_mlflow()}` (definido em `config/modeling.yaml`).")

cabecalho_pagina(
    "Comparação entre modelos experimentados",
    "Cada linha = uma combinação algoritmo × redução. Se treinou várias vezes, mostra-se só o run **mais recente** por combinação.",
)

try:
    tabela = tabela_comparacao(db_uri)
except Exception as exc:
    tabela = None
    st.error(f"Não foi possível ler o MLflow: {exc}")

if tabela is None:
    pass
elif tabela.empty:
    st.warning(
        "Nenhum run encontrado. Execute `python notebooks/modelagem.py` na raiz do projeto "
        "para popular o experimento e o ficheiro `mlruns.db`."
    )
else:
    st.markdown("**Tabela comparativa** (uma linha por modelo × redução; ordenada por F1 no teste)")
    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True,
        column_config={
            "F1 (teste)": st.column_config.NumberColumn(format="%.4f"),
            "Acurácia (teste)": st.column_config.NumberColumn(format="%.4f"),
            "ROC-AUC (teste)": st.column_config.NumberColumn(format="%.4f"),
            "Precisão (teste)": st.column_config.NumberColumn(format="%.4f"),
            "Revocação (teste)": st.column_config.NumberColumn(format="%.4f"),
            "Treino (s)": st.column_config.NumberColumn(format="%.2f"),
            "Inferência teste (s)": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.divider()
    st.markdown(
        "**F1 no teste por configuração** — eixo fixo entre 0 e 1 (melhores mais à direita; no gráfico, piores em baixo)."
    )
    graf = dataframe_grafico_f1(db_uri)
    if not graf.empty:
        df_plot = graf.sort_values("F1 (teste)", ascending=True).copy()
        df_plot["F1 (teste)"] = df_plot["F1 (teste)"].clip(lower=0.0, upper=1.0)
        n = len(df_plot)
        fig, ax = plt.subplots(figsize=(9, max(3.6, 0.42 * n)))
        y_pos = range(n)
        ax.barh(
            list(y_pos),
            df_plot["F1 (teste)"].values,
            height=0.65,
            color="#1a4a7a",
            edgecolor="#0f2744",
            linewidth=0.35,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(list(df_plot.index), fontsize=9)
        ax.set_xlabel("F1 (conjunto de teste)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.5, n - 0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.caption(
        "O modelo servido na predição é o registado como `wine-quality-opinion-best` "
        "(melhor F1 na execução de `modelagem.py`), "
        "que pode não coincidir com a primeira linha se tiver voltado a treinar depois."
    )

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/1_Predicao.py", label="Predição", icon="📋")
with c2:
    st.page_link("pages/3_Monitoramento.py", label="Monitoramento", icon="📈")
