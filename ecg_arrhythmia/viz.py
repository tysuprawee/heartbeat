"""Qualitative visualization: plot a 12-lead ECG with detected R-peaks."""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, data
from .features import detect_rpeaks
from .preprocess import preprocess_record


def plot_record(ecg_id: int, raw: np.ndarray, save_path) -> None:
    """Plot all 12 leads of one record; overlay R-peaks on lead II."""
    proc = preprocess_record(raw)
    t = np.arange(config.SIGNAL_LEN) / config.FS
    peaks = detect_rpeaks(proc[:, config.LEAD_II])

    fig, axes = plt.subplots(6, 2, figsize=(13, 11), sharex=True)
    for li, ax in enumerate(axes.T.flatten()):
        ax.plot(t, proc[:, li], lw=0.8, color="#1a1a1a")
        ax.set_ylabel(config.LEADS[li], rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.25)
        if li == config.LEAD_II:
            ax.plot(t[peaks], proc[peaks, li], "rx", ms=7, label="R-peak")
            ax.legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"PTB-XL record {ecg_id}: preprocessed 12-lead ECG "
                 f"({len(peaks)} beats, ~{len(peaks) * 6} bpm)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved ECG figure -> {save_path}")


def plot_record_by_id(ecg_id: int, save_path=None):
    """Load a single record by ecg_id and plot it."""
    config.ensure_dirs()
    labels = data.build_label_table()
    if ecg_id not in labels.index:
        raise SystemExit(f"ecg_id {ecg_id} not found in label table")
    fn = labels.loc[ecg_id, "filename_lr"]
    raw = data.load_signals([fn])[0]
    save_path = save_path or (config.FIGURES_DIR / "ecg_sample.png")
    plot_record(ecg_id, raw, save_path)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Plot a 12-lead ECG record.")
    ap.add_argument("--ecg-id", type=int, default=1)
    args = ap.parse_args()
    plot_record_by_id(args.ecg_id)


if __name__ == "__main__":
    _main()
