"""Evaluation: held-out metrics, Matplotlib figures, and REPORT.md.

Scores the frozen models on the official test fold (10), writes all figures to
reports/figures/, and generates REPORT.md with a dedicated analysis section
per figure (numbers are filled in from the computed metrics).
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)

from . import config, data
from .train import _proba

FIG = config.FIGURES_DIR


# --- shared loading --------------------------------------------------------
def _load():
    feats = pd.read_parquet(config.FEATURES_CACHE)
    labels = pd.read_parquet(config.LABELS_CACHE)
    idx = data.split_indices(labels)["test"]
    return feats, labels, idx


def _feature_importances(model, feature_names) -> pd.Series | None:
    """Best-effort importance vector from the winning pipeline."""
    clf = model.named_steps["clf"]
    ests = getattr(clf, "estimators_", None)

    def _from(est):
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_)
        if hasattr(est, "coef_"):
            return np.abs(np.asarray(est.coef_)).reshape(-1, len(feature_names)).mean(0)
        return None

    if ests is not None:                       # multilabel: average across outputs
        vals = [_from(e) for e in ests]
        vals = [v for v in vals if v is not None]
        imp = np.mean(vals, axis=0) if vals else None
    else:
        imp = _from(clf)
    return pd.Series(imp, index=feature_names) if imp is not None else None


# --- EDA figure ------------------------------------------------------------
def fig_class_prevalence(labels: pd.DataFrame) -> dict:
    prev = {sc: float(labels[sc].mean()) for sc in config.SUPERCLASSES}
    binbal = float(labels["binary"].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(prev.keys(), [v * 100 for v in prev.values()], color="#4878CF")
    ax1.set_title("Diagnostic superclass prevalence")
    ax1.set_ylabel("% of records (multi-label)")
    for i, v in enumerate(prev.values()):
        ax1.text(i, v * 100 + 0.5, f"{v:.0%}", ha="center", fontsize=9)

    ax2.bar(["normal", "abnormal"], [(1 - binbal) * 100, binbal * 100],
            color=["#5BA053", "#C44E52"])
    ax2.set_title("Binary class balance")
    ax2.set_ylabel("% of records")
    for i, v in enumerate([1 - binbal, binbal]):
        ax2.text(i, v * 100 + 0.5, f"{v:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "eda_class_prevalence.png", dpi=120)
    plt.close(fig)
    return {"prevalence": prev, "binary_abnormal": binbal}


# --- multilabel figures + metrics -----------------------------------------
def eval_multilabel(feats, labels, test_idx) -> dict:
    bundle = joblib.load(config.MODELS_DIR / "multilabel_model.joblib")
    model = bundle["model"]
    X = feats.to_numpy()[test_idx]
    Y = labels[config.SUPERCLASSES].to_numpy()[test_idx]
    scores = _proba(model, X)
    preds = (scores >= 0.5).astype(int)

    per_auc = {sc: float(roc_auc_score(Y[:, i], scores[:, i]))
               for i, sc in enumerate(config.SUPERCLASSES)}
    per_f1 = {sc: float(f1_score(Y[:, i], preds[:, i], zero_division=0))
              for i, sc in enumerate(config.SUPERCLASSES)}
    macro_auc = float(roc_auc_score(Y, scores, average="macro"))
    micro_auc = float(roc_auc_score(Y, scores, average="micro"))

    # ROC overlay
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, sc in enumerate(config.SUPERCLASSES):
        fpr, tpr, _ = roc_curve(Y[:, i], scores[:, i])
        ax.plot(fpr, tpr, lw=1.8, label=f"{sc} (AUC {per_auc[sc]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"Per-superclass ROC (macro-AUC {macro_auc:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "multilabel_roc.png", dpi=120); plt.close(fig)

    # per-class AUC + F1 bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(config.SUPERCLASSES)); w = 0.38
    ax.bar(x - w / 2, [per_auc[s] for s in config.SUPERCLASSES], w, label="ROC-AUC", color="#4878CF")
    ax.bar(x + w / 2, [per_f1[s] for s in config.SUPERCLASSES], w, label="F1", color="#EE854A")
    ax.set_xticks(x); ax.set_xticklabels(config.SUPERCLASSES)
    ax.set_ylim(0, 1); ax.set_title("Per-superclass AUC and F1 (test fold)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "multilabel_per_class.png", dpi=120); plt.close(fig)

    per_acc = {sc: float((preds[:, i] == Y[:, i]).mean())
               for i, sc in enumerate(config.SUPERCLASSES)}
    subset_acc = float((preds == Y).all(axis=1).mean())
    hamming_acc = float((preds == Y).mean())

    imp = _feature_importances(model, bundle["feature_names"])
    return {"model": bundle["name"], "macro_auc": macro_auc, "micro_auc": micro_auc,
            "per_auc": per_auc, "per_f1": per_f1, "per_acc": per_acc,
            "subset_acc": subset_acc, "hamming_acc": hamming_acc,
            "importances": imp, "n_test": len(test_idx)}


# --- binary figures + metrics ---------------------------------------------
def eval_binary(feats, labels, test_idx) -> dict:
    bundle = joblib.load(config.MODELS_DIR / "binary_model.joblib")
    model = bundle["model"]
    X = feats.to_numpy()[test_idx]
    y = labels["binary"].to_numpy()[test_idx]
    score = _proba(model, X)
    pred = (score >= 0.5).astype(int)

    auc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    f1 = float(f1_score(y, pred))
    acc = float((pred == y).mean())
    cm = confusion_matrix(y, pred)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal", "abnormal"])
    ax.set_yticks([0, 1], labels=["normal", "abnormal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Binary confusion matrix (acc {acc:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG / "binary_confusion.png", dpi=120); plt.close(fig)

    # ROC + PR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y, score)
    ax1.plot(fpr, tpr, lw=2, color="#C44E52", label=f"AUC {auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax1.set(xlabel="FPR", ylabel="TPR", title="Binary ROC"); ax1.legend(loc="lower right")
    prec, rec, _ = precision_recall_curve(y, score)
    ax2.plot(rec, prec, lw=2, color="#4878CF", label=f"AP {ap:.3f}")
    ax2.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"baseline {y.mean():.2f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Binary precision-recall")
    ax2.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "binary_roc_pr.png", dpi=120); plt.close(fig)

    imp = _feature_importances(model, bundle["feature_names"])
    return {"model": bundle["name"], "auc": auc, "ap": ap, "f1": f1, "acc": acc,
            "cm": cm.tolist(), "importances": imp, "n_test": len(test_idx)}


# --- feature importance figure (shared) -----------------------------------
def fig_feature_importance(ml: dict, bi: dict, feats, labels, test_idx) -> list[str]:
    imp = ml.get("importances")
    src = "multi-label"
    if imp is None:
        imp = bi.get("importances"); src = "binary"
    if imp is None:
        # Winning model (e.g. HistGBDT) exposes no native importances: fall back
        # to model-agnostic permutation importance on the binary task.
        from sklearn.inspection import permutation_importance
        bundle = joblib.load(config.MODELS_DIR / "binary_model.joblib")
        X = feats.to_numpy()[test_idx]
        y = labels["binary"].to_numpy()[test_idx]
        r = permutation_importance(bundle["model"], X, y, n_repeats=5,
                                   scoring="roc_auc", random_state=config.RANDOM_STATE,
                                   n_jobs=-1)
        imp = pd.Series(r.importances_mean, index=bundle["feature_names"])
        src = "binary, permutation"
    if imp is None:
        return []
    top = imp.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(top.index[::-1], top.values[::-1], color="#6ACC64")
    ax.set_title(f"Top-20 feature importances ({src} model)")
    ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=120); plt.close(fig)
    return list(top.index)


# --- report generation -----------------------------------------------------
def write_report(eda, ml, bi, top_feats, labels):
    """Build the metric context and render every language version."""
    from . import report_templates as rt

    cm = bi["cm"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    idx = data.split_indices(labels)
    n_abn = int((labels["binary"] == 1).sum())
    ctx = {
        "model_ml": ml["model"], "model_bi": bi["model"],
        "macro_auc": ml["macro_auc"], "micro_auc": ml["micro_auc"],
        "bi_auc": bi["auc"], "bi_ap": bi["ap"], "bi_f1": bi["f1"], "bi_acc": bi["acc"],
        "hamming_acc": ml["hamming_acc"], "subset_acc": ml["subset_acc"],
        "per_auc": ml["per_auc"], "per_f1": ml["per_f1"], "per_acc": ml["per_acc"],
        "prevalence": eda["prevalence"], "binary_abnormal": eda["binary_abnormal"],
        "counts": {sc: int(labels[sc].sum()) for sc in config.SUPERCLASSES},
        "n_total": len(labels), "n_abn": n_abn, "n_norm": len(labels) - n_abn,
        "n_tr": len(idx["train"]), "n_va": len(idx["val"]), "n_te": len(idx["test"]),
        "n_test": ml["n_test"],
        "best_sc": max(ml["per_auc"], key=ml["per_auc"].get),
        "worst_sc": min(ml["per_auc"], key=ml["per_auc"].get),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "sens": tp / (tp + fn) if (tp + fn) else 0.0,
        "spec": tn / (tn + fp) if (tn + fp) else 0.0,
        "top_feats": top_feats,
    }
    for filename, render in rt.RENDERERS.items():
        out = config.PROJECT_ROOT / filename
        out.write_text(render(ctx), encoding="utf-8")
        print(f"Wrote report -> {out}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate models and build report.")
    ap.add_argument("--task", choices=["multilabel", "binary", "both"], default="both")
    ap.add_argument("--ecg-id", type=int, default=1, help="record for the sample ECG figure")
    args = ap.parse_args()

    config.ensure_dirs()
    feats, labels, test_idx = _load()
    eda = fig_class_prevalence(labels)

    # sample ECG figure for the report
    from .viz import plot_record_by_id
    plot_record_by_id(args.ecg_id)

    ml = eval_multilabel(feats, labels, test_idx)
    bi = eval_binary(feats, labels, test_idx)
    print(f"multilabel macro-AUC = {ml['macro_auc']:.4f} | binary AUC = {bi['auc']:.4f}")

    top = fig_feature_importance(ml, bi, feats, labels, test_idx)
    write_report(eda, ml, bi, top, labels)


if __name__ == "__main__":
    _main()
