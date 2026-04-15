# %%
"""
Parte 4 — t-SNE para visualização e interpretabilidade.
t-SNE não expõe transform() estável para novos pontos no sklearn; usamos
apenas para diagnóstico 2D (enunciado: comparar com PCA no relatório).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config_loader import load_yaml

# %%
pipe = load_yaml(ROOT_DIR / "config" / "pipeline.yaml")
feat_path = ROOT_DIR / pipe["paths"]["features_data_dir"] / pipe["paths"]["features_filename"]
df = pd.read_parquet(feat_path)
y = df["opinion"].values
X = df.drop(columns=["opinion"]).values

X_scaled = RobustScaler().fit_transform(X)
emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)

out_dir = ROOT_DIR / "outputs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7, 6))
plt.scatter(emb[y == 0, 0], emb[y == 0, 1], alpha=0.35, label="opinion=0 (quality≤5)")
plt.scatter(emb[y == 1, 0], emb[y == 1, 1], alpha=0.35, label="opinion=1 (quality>5)")
plt.legend()
plt.title("t-SNE (2D) — vinho branco")
plt.tight_layout()
fp = out_dir / "tsne_opinion_2d.png"
plt.savefig(fp, dpi=150)
plt.close()
print("Figura salva:", fp)
