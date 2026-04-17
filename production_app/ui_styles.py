"""Estilo visual comum às páginas (layout tipo dashboard acadêmico)."""

from __future__ import annotations

import streamlit as st


def aplicar_estilo() -> None:
    st.markdown(
        """
        <style>
        /* Barra superior mais sóbria */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #0f2744 0%, #1a4a7a 100%);
        }
        header[data-testid="stHeader"] * {
            color: #f0f4f8 !important;
        }
        /* Títulos de página */
        h1 { font-weight: 600 !important; letter-spacing: -0.02em; }
        h2, h3 { font-weight: 600 !important; color: #0f2744 !important; }
        /* Cards de métrica */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e6ef;
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] > div {
            background: #ffffff;
            border-right: 1px solid #e2e6ef;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def cabecalho_pagina(titulo: str, subtitulo: str | None = None) -> None:
    st.title(titulo)
    if subtitulo:
        st.caption(subtitulo)
