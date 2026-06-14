"""Feature extraction: the biomedical signal-processing core.

For each record we compute a fixed-length feature vector combining:

* rhythm / HRV features from R-peak detection on lead II,
* per-lead time-domain statistics, and
* per-lead spectral band powers (Welch PSD).

Everything is NumPy/SciPy. Records are preprocessed (filtered + normalized)
before features are computed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew

from . import config
from .preprocess import preprocess_record

# Minimum spacing between R-peaks: ~250 ms refractory period (240 bpm ceiling).
_MIN_RR_SAMPLES = int(0.25 * config.FS)


# --- R-peak / HRV ----------------------------------------------------------
def detect_rpeaks(lead_ii: np.ndarray) -> np.ndarray:
    """Detect R-peaks on a normalized lead-II signal.

    The signal is z-scored, so a height threshold in standard-deviation units
    is robust across records. Returns peak sample indices.
    """
    peaks, _ = find_peaks(lead_ii, height=1.0, distance=_MIN_RR_SAMPLES)
    return peaks


def hrv_features(peaks: np.ndarray) -> dict[str, float]:
    """Heart-rate / HRV features derived from R-peak indices."""
    if len(peaks) < 3:
        # Not enough beats to characterize rhythm: return neutral defaults.
        return {
            "hr_mean": 0.0, "rr_mean": 0.0, "rr_std": 0.0, "rr_min": 0.0,
            "rr_max": 0.0, "sdnn": 0.0, "rmssd": 0.0, "n_beats": float(len(peaks)),
        }
    rr = np.diff(peaks) / config.FS  # RR intervals in seconds
    hr = 60.0 / rr
    diff_rr = np.diff(rr)
    return {
        "hr_mean": float(np.mean(hr)),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "rr_min": float(np.min(rr)),
        "rr_max": float(np.max(rr)),
        "sdnn": float(np.std(rr)),
        "rmssd": float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) else 0.0,
        "n_beats": float(len(peaks)),
    }


# --- Per-lead time-domain stats -------------------------------------------
def lead_time_features(lead: np.ndarray) -> dict[str, float]:
    """Time-domain statistics for a single lead."""
    return {
        "std": float(np.std(lead)),
        "rms": float(np.sqrt(np.mean(lead ** 2))),
        "range": float(np.ptp(lead)),
        "skew": float(skew(lead)),
        "kurtosis": float(kurtosis(lead)),
        "zcr": float(np.mean(np.abs(np.diff(np.sign(lead))) > 0)),
    }


# --- Per-lead spectral features -------------------------------------------
def lead_spectral_features(lead: np.ndarray) -> dict[str, float]:
    """Welch band-power features for a single lead."""
    freqs, psd = welch(lead, fs=config.FS, nperseg=min(256, len(lead)))
    total = trapezoid(psd, freqs) + 1e-12
    feats = {}
    for name, (lo, hi) in config.FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"bp_{name}"] = float(trapezoid(psd[mask], freqs[mask]) / total)
    return feats


# --- Record-level assembly -------------------------------------------------
def extract_record(record: np.ndarray) -> dict[str, float]:
    """Compute the full feature dict for one raw (T, leads) record."""
    proc = preprocess_record(record)
    feats: dict[str, float] = {}

    # Rhythm features from lead II.
    peaks = detect_rpeaks(proc[:, config.LEAD_II])
    feats.update(hrv_features(peaks))

    # Per-lead time + spectral features.
    for li, lead_name in enumerate(config.LEADS):
        lead = proc[:, li]
        for k, v in lead_time_features(lead).items():
            feats[f"{lead_name}_{k}"] = v
        for k, v in lead_spectral_features(lead).items():
            feats[f"{lead_name}_{k}"] = v
    return feats


def build_feature_matrix(signals: np.ndarray) -> pd.DataFrame:
    """Extract features for every record into a (N, F) DataFrame."""
    from tqdm import tqdm

    rows = [extract_record(np.asarray(signals[i])) for i in
            tqdm(range(len(signals)), desc="Extracting features")]
    df = pd.DataFrame(rows).astype(np.float32)
    # R-peak failures already return neutral defaults, but guard anyway.
    return df.fillna(0.0)


def _main() -> None:
    from . import data

    ap = argparse.ArgumentParser(description="Build PTB-XL feature matrix.")
    ap.add_argument("--sample", type=int, default=None, help="use only first N records")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.sample is not None:
        signals, labels = data.build_cache(sample=args.sample)
    else:
        signals, labels = data.load_cache()

    feats = build_feature_matrix(signals)
    feats.index = labels.index
    if args.sample is None:
        feats.to_parquet(config.FEATURES_CACHE)
        print(f"Cached features -> {config.FEATURES_CACHE} {feats.shape}")
    else:
        print(f"Built features (not cached, sample run): {feats.shape}")


if __name__ == "__main__":
    _main()
