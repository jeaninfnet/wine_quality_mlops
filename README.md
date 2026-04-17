# Wine Quality — MLOps

Projeto estruturado em torno do **mesmo problema do Projeto 1**: classificação binária `opinion` em **vinho branco** (dataset [rajyellow46/wine-quality](https://www.kaggle.com/datasets/rajyellow46/wine-quality) no Kaggle).

## Visão das etapas

1. **Ingestão** — download Kaggle + CSV → Parquet (`config/data.yaml`).
2. **Qualidade** — Validação dos dados (`config/quality.yaml`).
3. **Pré-processamento** — constrói `opinion`, filtra branco, remove `quality` (`src/wine_pipeline/build_features.py`).
4. **Modelagem** — `RandomizedSearchCV` + pipelines (RobustScaler → SelectKBest → PCA|LDA|none → classificador) + **MLflow** (SQLite `mlruns.db`).
5. **t-SNE** — apenas visualização: `notebooks/reducao_tsne.py`.
6. **App** — Streamlit: predição, **comparação de runs** (MLflow) e monitoramento simulado.

## Pré-requisitos

- Python 3.10+
- Conta Kaggle + `secrets.env` na raiz (veja `secrets.env.example`).

## Comandos

```bash
cd wine_quality_mlops
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy secrets.env.example secrets.env
# edite secrets.env com KAGGLE_USERNAME e KAGGLE_KEY

python notebooks/ingestao.py
python notebooks/qualidade.py
python notebooks/preprocessamento.py
python notebooks/modelagem.py
python notebooks/reducao_tsne.py

mlflow ui --backend-store-uri sqlite:///mlruns.db  

streamlit run production_app/app.py
```

## Estrutura

- `config/` — políticas (YAML).
- `src/` — ingestão, qualidade, features, modelagem.
- `notebooks/` — orquestração fina (scripts com células `# %%`).
- `production_app/` — Streamlit.

## Notas para o relatório (Parte 4)

- **PCA** e **LDA** estão integrados ao `Pipeline` sklearn (`src/modeling/reducer.py`).