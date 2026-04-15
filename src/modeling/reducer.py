"""
Redução de dimensionalidade para integração em Pipeline sklearn.
Métodos: none | pca | lda (Parte 4 do enunciado).
t-SNE fica em notebooks/reducao_tsne.py (visualização — sem transform em novos dados).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class FeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method: str = "none",
        n_components: int | None = 5,
        random_state: int = 42,
        logger: Any = None,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.logger = logger
        self.reducer_: Any = None
        self.feature_names_in_: list[str] | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None

        if self.method == "none":
            self.reducer_ = None
            self.feature_names_out_ = self.feature_names_in_
            return self

        n_feat = X.shape[1] if hasattr(X, "shape") else len(self.feature_names_in_ or [])

        if self.method == "pca":
            k = self.n_components or min(n_feat, X.shape[0]) - 1
            k = max(1, min(k, n_feat))
            self.reducer_ = PCA(n_components=k, random_state=self.random_state)
            self.reducer_.fit(X.values if isinstance(X, pd.DataFrame) else X)
            self.feature_names_out_ = [f"pca_{i}" for i in range(self.reducer_.n_components_)]
            return self

        if self.method == "lda":
            if y is None:
                raise ValueError("LDA requer y no Pipeline.")
            k = self.n_components
            max_c = min(n_feat, len(np.unique(y)) - 1)
            k = max(1, min(k or max_c, max_c))
            self.reducer_ = LinearDiscriminantAnalysis(n_components=k)
            self.reducer_.fit(X.values if isinstance(X, pd.DataFrame) else X, np.ravel(y))
            self.feature_names_out_ = [f"lda_{i}" for i in range(k)]
            return self

        raise ValueError(f"method desconhecido: {self.method}")

    def transform(self, X, y=None):
        if self.reducer_ is None:
            return X
        Xa = X.values if isinstance(X, pd.DataFrame) else X
        Xt = self.reducer_.transform(Xa)
        cols = self.feature_names_out_ or [f"out_{i}" for i in range(Xt.shape[1])]
        idx = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(Xt, columns=cols, index=idx)
