"""
Orquestra experimentos (Partes 3–5): Pipeline sklearn + CV + MLflow.
Compara redução: none | pca | lda (Parte 4).
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.modeling.reducer import FeatureReducer
from src.utils.config_loader import load_yaml


def _import_class(path: str):
    mod_name, _, cls_name = path.rpartition(".")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _make_estimator(spec: dict[str, Any], search: dict[str, Any]):
    cls = _import_class(spec["estimator"])
    params = {**spec.get("params", {})}
    return cls(**params), search


def _salvar_figuras_modelagem(
    root_dir: Path,
    run_name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
) -> None:
    """Grava PNGs em outputs/modeling/<run_name>/ para o relatório e anexa ao run MLflow."""
    out_dir = root_dir / "outputs" / "modeling" / run_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        display_labels=["0 (≤5)", "1 (>5)"],
    )
    ax.set_title("Matriz de confusão (holdout)")
    ax.set_xlabel("Classe predita pelo modelo")
    ax.set_ylabel("Classe real (opinion)")
    fig.text(
        0.5,
        0.02,
        "Cada célula: quantas amostras (real na linha, predito na coluna).",
        ha="center",
        fontsize=8,
        style="italic",
    )
    p_cm = out_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p_cm, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(p_cm))

    if proba is not None and len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)

        # ROC: cada ponto = (FPR, TPR) ao variar o limiar de corte da probabilidade
        fig2, ax2 = plt.subplots(figsize=(5.8, 5))
        sc = ax2.scatter(
            fpr,
            tpr,
            s=32,
            c=range(len(fpr)),
            cmap="viridis",
            zorder=5,
            edgecolors="white",
            linewidths=0.45,
            label="pontos (limiares)",
        )
        ax2.plot(fpr, tpr, "-", color="C0", linewidth=1.0, alpha=0.75, zorder=4)
        cbar = fig2.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Ordem ao longo da curva\n(0≈limiar alto, último≈baixo)", fontsize=7)
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.35, label="aleatório")
        ax2.set_xlabel("Taxa de falsos positivos (FPR)")
        ax2.set_ylabel("Taxa de verdadeiros positivos (TPR)")
        ax2.set_title(f"Curva ROC (holdout) — AUC = {auc:.3f}")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        fig2.text(
            0.5,
            0.01,
            "Cada ponto não é um vinho: é um par (FPR,TPR) ao mudar o limiar de P(opinion=1).",
            ha="center",
            fontsize=7,
            style="italic",
        )
        fig2.tight_layout()
        fig2.subplots_adjust(bottom=0.14)
        p_roc = out_dir / "roc_curve.png"
        fig2.savefig(p_roc, dpi=120)
        plt.close(fig2)
        mlflow.log_artifact(str(p_roc))

        # Probabilidade P(opinion=1) por classe real — nuvens de pontos separadas (jitter no eixo x)
        m0 = y_test == 0
        m1 = y_test == 1
        x0 = rng.uniform(-0.12, 0.12, size=int(m0.sum()))
        x1 = 1.0 + rng.uniform(-0.12, 0.12, size=int(m1.sum()))
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        ax3.scatter(
            x0,
            proba[m0],
            alpha=0.55,
            s=22,
            c="tab:blue",
            edgecolors="white",
            linewidths=0.3,
            label="opinion real = 0",
        )
        ax3.scatter(
            x1,
            proba[m1],
            alpha=0.55,
            s=22,
            c="tab:orange",
            edgecolors="white",
            linewidths=0.3,
            label="opinion real = 1",
        )
        ax3.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="limiar 0,5")
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["vinhos com opinion real = 0", "vinhos com opinion real = 1"])
        ax3.set_ylabel("Probabilidade predita P(opinion = 1)")
        ax3.set_title("Cada ponto = um vinho do holdout (cor = classe real)")
        ax3.legend(loc="best", fontsize=9)
        ax3.set_xlim(-0.35, 1.35)
        fig3.text(
            0.5,
            0.01,
            "Eixo horizontal só separa os dois grupos reais; o jitter evita sobreposição.",
            ha="center",
            fontsize=7,
            style="italic",
        )
        fig3.tight_layout()
        fig3.subplots_adjust(bottom=0.12)
        p_sc = out_dir / "proba_por_classe_real.png"
        fig3.savefig(p_sc, dpi=120)
        plt.close(fig3)
        mlflow.log_artifact(str(p_sc))

        # Um ponto por amostra: (real, predito); verde = acerto, vermelho = erro
        jitter_x = rng.uniform(-0.08, 0.08, size=len(y_test))
        jitter_y = rng.uniform(-0.08, 0.08, size=len(y_test))
        fig4, ax4 = plt.subplots(figsize=(5.2, 5.2))
        acerto = y_test == y_pred
        ax4.scatter(
            y_test[acerto].astype(float) + jitter_x[acerto],
            y_pred[acerto].astype(float) + jitter_y[acerto],
            alpha=0.5,
            s=22,
            c="tab:green",
            edgecolors="white",
            linewidths=0.25,
            label="acerto",
        )
        ax4.scatter(
            y_test[~acerto].astype(float) + jitter_x[~acerto],
            y_pred[~acerto].astype(float) + jitter_y[~acerto],
            alpha=0.65,
            s=26,
            c="tab:red",
            edgecolors="white",
            linewidths=0.25,
            label="erro",
        )
        ax4.plot([-0.2, 1.2], [-0.2, 1.2], "k--", alpha=0.35, label="diagonal perfeita")
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xlabel("opinion real (0 ou 1)")
        ax4.set_ylabel("opinion predita (0 ou 1)")
        ax4.set_title("Cada ponto = um vinho: comparação real × predito")
        ax4.set_xlim(-0.25, 1.25)
        ax4.set_ylim(-0.25, 1.25)
        ax4.legend(loc="upper left", fontsize=9)
        fig4.text(
            0.5,
            0.02,
            "Pontos na diagonal tracejada = classificação correta.",
            ha="center",
            fontsize=7,
            style="italic",
        )
        fig4.tight_layout()
        fig4.subplots_adjust(bottom=0.1)
        p_rp = out_dir / "real_vs_predito.png"
        fig4.savefig(p_rp, dpi=120)
        plt.close(fig4)
        mlflow.log_artifact(str(p_rp))


def run_classification_experiments(root_dir: Path, logger: Any = None) -> None:
    log = logger
    cfg = load_yaml(root_dir / "config" / "modeling.yaml")
    pipe_cfg = load_yaml(root_dir / "config" / "pipeline.yaml")
    features_path = root_dir / pipe_cfg["paths"]["features_data_dir"] / pipe_cfg["paths"]["features_filename"]

    df = pd.read_parquet(features_path)
    target = cfg["experiment"]["target_column"]
    y = df[target].values.ravel()
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["experiment"]["test_size"],
        stratify=y,
        random_state=cfg["experiment"]["random_state"],
    )

    uri = "sqlite:///" + str((root_dir / "mlruns.db").resolve()).replace("\\", "/")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["experiment"]["name"])

    k_kbest = min(cfg["feature_engineering"]["select_k_best"]["k"], X.shape[1])
    n_comp = cfg["reduction"]["n_components"]
    reductions = cfg["reduction"]["methods_compare"]
    cv = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=True,
        random_state=cfg["experiment"]["random_state"],
    )

    best_f1 = -1.0
    best_run_id: str | None = None

    for model_key, spec in cfg["models"].items():
        for red in reductions:
            est, _search = _make_estimator(spec, spec["search"])
            run_name = f"{model_key}__reduction_{red}"
            with mlflow.start_run(run_name=run_name):
                reducer = FeatureReducer(method=red, n_components=n_comp, random_state=42)
                base_pipe = Pipeline(
                    [
                        ("scaler", RobustScaler()),
                        ("kbest", SelectKBest(f_classif, k=k_kbest)),
                        ("reducer", reducer),
                        ("clf", est),
                    ]
                )

                param_dist = {f"clf__{k}": v for k, v in spec["search"].items()}
                search_cv = RandomizedSearchCV(
                    base_pipe,
                    param_distributions=param_dist,
                    n_iter=18,
                    scoring=cfg["cv"]["scoring"],
                    cv=cv,
                    random_state=42,
                    n_jobs=-1,
                    refit=True,
                )

                t0 = time.perf_counter()
                search_cv.fit(X_train, y_train)
                train_s = time.perf_counter() - t0

                best: Pipeline = search_cv.best_estimator_
                t1 = time.perf_counter()
                y_pred = best.predict(X_test)
                infer_s = time.perf_counter() - t1

                proba = None
                if hasattr(best.named_steps["clf"], "predict_proba"):
                    proba = best.predict_proba(X_test)[:, 1]
                elif hasattr(best.named_steps["clf"], "decision_function"):
                    proba = best.decision_function(X_test)

                f1 = f1_score(y_test, y_pred)
                mlflow.log_params(
                    {
                        "model": model_key,
                        "reduction": red,
                        **{f"best__{k}": v for k, v in search_cv.best_params_.items()},
                    }
                )
                mlflow.log_metrics(
                    {
                        "f1_test": f1,
                        "accuracy_test": accuracy_score(y_test, y_pred),
                        "precision_test": precision_score(y_test, y_pred, zero_division=0),
                        "recall_test": recall_score(y_test, y_pred, zero_division=0),
                        "train_seconds": train_s,
                        "infer_seconds_test_set": infer_s,
                    }
                )
                if proba is not None:
                    try:
                        mlflow.log_metric("roc_auc_test", roc_auc_score(y_test, proba))
                    except Exception:
                        pass

                _salvar_figuras_modelagem(root_dir, run_name, y_test, y_pred, proba)

                mlflow.sklearn.log_model(best, artifact_path="model")

                if f1 > best_f1:
                    best_f1 = f1
                    best_run_id = mlflow.active_run().info.run_id

    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        name = cfg["mlflow"]["registry_model_name"]
        try:
            mlflow.register_model(model_uri, name)
            if log:
                log.info("Modelo registrado: models:/%s/latest (run %s)", name, best_run_id)
        except Exception as exc:
            if log:
                log.warning("Registro MLflow (pode já existir): %s", exc)
