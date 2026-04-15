# Parte 1 — Mapeamento dos experimentos (Projeto 1 → Projeto 2)

Este documento amarra o trabalho anterior (`trabalho_jean`) ao pipeline reprodutível deste repositório.

## Problema de negócio

Classificar se um **vinho branco** tende a ter **qualidade percebida > 5** no score original (`opinion = 1`), a partir de atributos físico-químicos.

## Dataset

- **Fonte:** Kaggle `rajyellow46/wine-quality`, arquivo `winequalityN.csv`.
- **Escopo modelagem:** apenas linhas `type == white` (como no Projeto 1).
- **Alvo:** `opinion = 1` se `quality > 5`, senão `0`. A coluna `quality` é removida antes da modelagem.

## Experimentos já realizados (Projeto 1)

| ID | Algoritmo | Pré-processamento | Validação | Métrica principal | Observação |
|----|-----------|-------------------|-----------|-------------------|------------|
| E1 | Regressão logística | `ColumnTransformer` (RobustScaler + OHE), `SelectKBest` | `StratifiedKFold(10)` + `RandomizedSearchCV` (70 iter) | F1 | Espaço amplo de `C`, `penalty`, `class_weight` |
| E2 | Árvore de decisão | Idem | Idem | F1 | `max_depth`, `criterion` |
| E3 | SVM (kernel default RBF no script; tuning só `k`) | Idem | Idem | F1 | Busca mais restrita em `k` |

## Critérios de sucesso (sugestão para o relatório)

- **Técnico:** F1 no holdout ≥ baseline (ex.: modelo constante ou regra simples).
- **Negócio:** reduzir falsos negativos em cenários onde “qualidade alta” importa (ajustar peso / recall conforme narrativa).
- **Operação:** pipeline roda de ponta a ponta com configs versionadas (`config/*.yaml`) e experimentos no MLflow.

## Riscos (engenharia de dados)

- Modelo treinado **só em brancos**: generalização para **tintos** é fraca (o Projeto 1 já mostrava isso) — covariate shift.
- `SelectKBest` depende de rótulo: deve permanecer **dentro** do `Pipeline` de treino (evitar leakage).

## O que mudou no Projeto 2

- Ingestão + qualidade + features materializadas em **Parquet**.
- Treino/validação em **módulos** + **MLflow** (métricas, parâmetros, artefatos).
- Comparação **PCA / LDA** no pipeline (Parte 4) + **t-SNE** em `notebooks/reducao_tsne.py` para visualização.
