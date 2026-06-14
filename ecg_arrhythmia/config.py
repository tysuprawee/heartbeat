"""Central configuration: paths, signal constants, label space, CV split."""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
DATABASE_CSV = DATA_ROOT / "ptbxl_database.csv"
SCP_CSV = DATA_ROOT / "scp_statements.csv"

CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SIGNALS_CACHE = CACHE_DIR / "signals_100.npy"
LABELS_CACHE = CACHE_DIR / "labels.parquet"
FEATURES_CACHE = CACHE_DIR / "features.parquet"

# --- Signal constants ------------------------------------------------------
FS = 100               # sampling rate of records100 (Hz)
SIGNAL_LEN = 1000      # samples per record (10 s @ 100 Hz)
N_LEADS = 12
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_II = 1            # index of lead II (used for R-peak detection)

# --- Label space -----------------------------------------------------------
# PTB-XL diagnostic superclasses (the official 5-class benchmark target).
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# --- Official stratified split (PTB-XL strat_fold 1..10) -------------------
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLD = 9
TEST_FOLD = 10

# --- Preprocessing ---------------------------------------------------------
BANDPASS_LO = 0.5      # Hz, removes baseline wander
BANDPASS_HI = 40.0     # Hz, removes HF/EMG noise (< 50 Hz Nyquist)
BANDPASS_ORDER = 4

# --- Spectral features -----------------------------------------------------
# Physiologically motivated bands for Welch band-power features (Hz).
FREQ_BANDS = {
    "vlf": (0.5, 3.0),
    "lf": (3.0, 10.0),
    "mf": (10.0, 20.0),
    "hf": (20.0, 40.0),
}

RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    for d in (CACHE_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
