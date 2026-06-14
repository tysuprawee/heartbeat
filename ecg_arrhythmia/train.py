"""Model training for both tasks using scikit-learn.

Features and labels come from the cached parquet/npy artifacts. Models are
fit on folds 1-8, selected on fold 9 (validation), and frozen. The held-out
fold 10 is never touched here -- it is scored in ``evaluate.py``.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data


def _load_features_labels(sample: int | None):
    """Return (features_df, labels_df) aligned by index."""
    if sample is not None:
        signals, labels = data.build_cache(sample=sample)
        from .features import build_feature_matrix
        feats = build_feature_matrix(signals)
        feats.index = labels.index
    else:
        feats = pd.read_parquet(config.FEATURES_CACHE)
        labels = pd.read_parquet(config.LABELS_CACHE)
    return feats, labels


def _candidate_models(task: str) -> dict[str, Pipeline]:
    """Pipelines (scaler + estimator) for the given task."""
    rs = config.RANDOM_STATE
    base = {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=rs, class_weight="balanced"),
        "hist_gbdt": HistGradientBoostingClassifier(random_state=rs),
    }
    models = {}
    for name, est in base.items():
        if task == "multilabel":
            # HistGBDT has no native multilabel support -> wrap per-output.
            wrapped = (MultiOutputClassifier(est) if name == "hist_gbdt"
                       else OneVsRestClassifier(est))
            models[name] = Pipeline([("scaler", StandardScaler()), ("clf", wrapped)])
        else:
            models[name] = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    return models


def _proba(model, X) -> np.ndarray:
    """Positive-class probabilities, shape (n,) binary or (n, k) multilabel."""
    p = model.predict_proba(X)
    if isinstance(p, list):                     # MultiOutputClassifier
        return np.column_stack([col[:, 1] for col in p])
    if p.ndim == 2 and p.shape[1] == 2:         # binary single output
        return p[:, 1]
    return p                                     # OneVsRest multilabel


def _score(y_true: np.ndarray, y_score: np.ndarray, task: str) -> float:
    """Selection metric: macro ROC-AUC (multilabel) or ROC-AUC (binary)."""
    if task == "multilabel":
        return roc_auc_score(y_true, y_score, average="macro")
    return roc_auc_score(y_true, y_score)


def train_task(task: str, feats: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Train all candidates for a task, select best on validation fold."""
    idx = data.split_indices(labels)
    X = feats.to_numpy()
    if task == "multilabel":
        Y = labels[config.SUPERCLASSES].to_numpy()
    else:
        Y = labels["binary"].to_numpy()

    Xtr, Ytr = X[idx["train"]], Y[idx["train"]]
    Xval, Yval = X[idx["val"]], Y[idx["val"]]

    results = {}
    best_name, best_score, best_model = None, -np.inf, None
    for name, model in _candidate_models(task).items():
        model.fit(Xtr, Ytr)
        val_score = _score(Yval, _proba(model, Xval), task)
        results[name] = val_score
        print(f"  [{task}] {name:14s} val-AUC = {val_score:.4f}")
        if val_score > best_score:
            best_name, best_score, best_model = name, val_score, model

    print(f"  [{task}] best = {best_name} (val-AUC {best_score:.4f})")
    out_path = config.MODELS_DIR / f"{task}_model.joblib"
    joblib.dump(
        {"model": best_model, "task": task, "name": best_name,
         "feature_names": list(feats.columns),
         "classes": config.SUPERCLASSES if task == "multilabel" else ["normal", "abnormal"]},
        out_path,
    )
    print(f"  [{task}] saved -> {out_path}")
    return {"val_scores": results, "best": best_name, "best_val_auc": best_score}


def _main() -> None:
    ap = argparse.ArgumentParser(description="Train ECG classifiers.")
    ap.add_argument("--task", choices=["multilabel", "binary", "both"], default="both")
    ap.add_argument("--sample", type=int, default=None, help="dev run on first N records")
    args = ap.parse_args()

    config.ensure_dirs()
    feats, labels = _load_features_labels(args.sample)
    print(f"Features {feats.shape}, labels {labels.shape}")

    tasks = ["multilabel", "binary"] if args.task == "both" else [args.task]
    for task in tasks:
        print(f"== Training: {task} ==")
        train_task(task, feats, labels)


if __name__ == "__main__":
    _main()
