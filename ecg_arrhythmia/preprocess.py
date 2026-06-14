"""Signal preprocessing with SciPy: bandpass filtering and normalization.

A single record is an (SIGNAL_LEN, N_LEADS) array. All functions are written
to operate column-wise so every lead is filtered independently.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from . import config

# Design the Butterworth bandpass once at import (SOS-free coefficient form is
# fine at order 4 / fs 100). 0.5 Hz removes baseline wander, 40 Hz removes
# high-frequency/EMG noise while staying below the 50 Hz Nyquist.
_NYQ = 0.5 * config.FS
_B, _A = butter(
    config.BANDPASS_ORDER,
    [config.BANDPASS_LO / _NYQ, config.BANDPASS_HI / _NYQ],
    btype="band",
)


def bandpass(record: np.ndarray) -> np.ndarray:
    """Zero-phase bandpass filter an (T, leads) record column-wise."""
    return filtfilt(_B, _A, record, axis=0).astype(np.float32)


def normalize(record: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalization (robust to flat leads)."""
    mean = record.mean(axis=0, keepdims=True)
    std = record.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((record - mean) / std).astype(np.float32)


def preprocess_record(record: np.ndarray) -> np.ndarray:
    """Full preprocessing chain for one record: bandpass then normalize."""
    return normalize(bandpass(record))
