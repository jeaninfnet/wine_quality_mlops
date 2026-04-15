import numpy as np
import pandas as pd

from src.modeling.reducer import FeatureReducer


def test_reducer_none_passthrough():
    X = pd.DataFrame(np.random.randn(20, 5), columns=list("abcde"))
    y = np.random.randint(0, 2, 20)
    r = FeatureReducer(method="none")
    r.fit(X, y)
    Xt = r.transform(X)
    assert Xt.shape == X.shape


def test_reducer_pca():
    X = pd.DataFrame(np.random.randn(50, 8), columns=[f"f{i}" for i in range(8)])
    y = np.random.randint(0, 2, 50)
    r = FeatureReducer(method="pca", n_components=3)
    r.fit(X, y)
    Xt = r.transform(X)
    assert Xt.shape == (50, 3)


def test_reducer_lda_binary():
    X = pd.DataFrame(np.random.randn(80, 6), columns=[f"f{i}" for i in range(6)])
    y = np.array([0] * 40 + [1] * 40)
    r = FeatureReducer(method="lda", n_components=1)
    r.fit(X, y)
    Xt = r.transform(X)
    assert Xt.shape == (80, 1)
