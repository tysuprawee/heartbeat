"""Data layer: metadata loading, label engineering, and signal caching.

The PTB-XL metadata (`ptbxl_database.csv`) gives one row per record with a
stringified ``scp_codes`` dict and a ``strat_fold`` for the official split.
``scp_statements.csv`` maps each SCP code to one of five diagnostic
superclasses. This module turns that into model-ready labels and a cached
signal array.
"""
from __future__ import annotations

import argparse
import ast
from typing import Iterable

import numpy as np
import pandas as pd

from . import config


# --- Metadata + label engineering -----------------------------------------
def load_metadata() -> pd.DataFrame:
    """Load ptbxl_database.csv with scp_codes parsed into real dicts."""
    df = pd.read_csv(config.DATABASE_CSV, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def build_code_to_superclass() -> dict[str, str]:
    """Map each diagnostic SCP code to its superclass (diagnostic_class)."""
    scp = pd.read_csv(config.SCP_CSV, index_col=0)
    scp = scp[scp["diagnostic"] == 1.0]
    return scp["diagnostic_class"].to_dict()


def aggregate_superclasses(
    scp_codes: dict[str, float], code_map: dict[str, str]
) -> list[str]:
    """Aggregate a record's SCP codes into the set of present superclasses.

    Follows the standard PTB-XL aggregation: a superclass is present if any of
    its diagnostic codes appears for the record (likelihood is ignored, as the
    benchmark does). Returns a sorted list for determinism.
    """
    classes = {code_map[c] for c in scp_codes if c in code_map}
    return sorted(classes)


def binary_label(superclasses: Iterable[str]) -> int:
    """0 = normal (only NORM present), 1 = abnormal (anything else)."""
    s = set(superclasses)
    return 0 if s == {"NORM"} else 1


def build_label_table() -> pd.DataFrame:
    """Build a per-record table of multi-label + binary targets and fold.

    Records with no diagnostic superclass are dropped (they cannot be scored
    against the 5-class benchmark); the dropped count is reported by the CLI.
    """
    df = load_metadata()
    code_map = build_code_to_superclass()

    superclasses = df["scp_codes"].apply(lambda c: aggregate_superclasses(c, code_map))
    keep = superclasses.apply(len) > 0
    df = df.loc[keep].copy()
    superclasses = superclasses.loc[keep]

    out = pd.DataFrame(index=df.index)
    out["strat_fold"] = df["strat_fold"]
    out["filename_lr"] = df["filename_lr"]
    # one binary column per superclass (multi-label target)
    for sc in config.SUPERCLASSES:
        out[sc] = superclasses.apply(lambda s, sc=sc: int(sc in s))
    out["binary"] = superclasses.apply(binary_label)
    return out


# --- Signal loading + cache ------------------------------------------------
def load_signals(filenames: Iterable[str]) -> np.ndarray:
    """Read WFDB records into an (N, SIGNAL_LEN, N_LEADS) float32 array."""
    import wfdb
    from tqdm import tqdm

    filenames = list(filenames)
    out = np.empty((len(filenames), config.SIGNAL_LEN, config.N_LEADS), dtype=np.float32)
    for i, fn in enumerate(tqdm(filenames, desc="Reading WFDB signals")):
        sig, _ = wfdb.rdsamp(str(config.DATA_ROOT / fn))
        out[i] = sig.astype(np.float32)
    return out


def build_cache(sample: int | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Build (and persist) the label table and signal array.

    With ``sample`` set, only the first N records are used (fast dev runs);
    the cache is written only for full runs to avoid clobbering it.
    """
    config.ensure_dirs()
    labels = build_label_table()
    if sample is not None:
        labels = labels.iloc[:sample].copy()

    signals = load_signals(labels["filename_lr"])

    if sample is None:
        np.save(config.SIGNALS_CACHE, signals)
        labels.to_parquet(config.LABELS_CACHE)
        print(f"Cached signals -> {config.SIGNALS_CACHE} {signals.shape}")
        print(f"Cached labels  -> {config.LABELS_CACHE} {labels.shape}")
    return signals, labels


def load_cache() -> tuple[np.ndarray, pd.DataFrame]:
    """Load the cached signal array (mem-mapped) and label table."""
    signals = np.load(config.SIGNALS_CACHE, mmap_mode="r")
    labels = pd.read_parquet(config.LABELS_CACHE)
    return signals, labels


def split_indices(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Positional train/val/test indices from the official strat_fold."""
    fold = labels["strat_fold"].to_numpy()
    return {
        "train": np.where(np.isin(fold, config.TRAIN_FOLDS))[0],
        "val": np.where(fold == config.VAL_FOLD)[0],
        "test": np.where(fold == config.TEST_FOLD)[0],
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Build PTB-XL label + signal cache.")
    ap.add_argument("--build-cache", action="store_true", help="build and persist cache")
    ap.add_argument("--sample", type=int, default=None, help="use only first N records")
    args = ap.parse_args()

    # Always report label statistics; build cache when requested.
    labels = build_label_table()
    total = len(load_metadata())
    print(f"Records with >=1 superclass: {len(labels)} / {total} "
          f"(dropped {total - len(labels)})")
    print("Superclass prevalence:")
    for sc in config.SUPERCLASSES:
        print(f"  {sc:5s}: {int(labels[sc].sum()):6d} ({labels[sc].mean():.1%})")
    print(f"Binary  abnormal: {int(labels['binary'].sum())} ({labels['binary'].mean():.1%})")

    if args.build_cache:
        build_cache(sample=args.sample)


if __name__ == "__main__":
    _main()
