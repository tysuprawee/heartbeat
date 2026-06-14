"""Unit tests for label engineering and the preprocessing/feature shapes."""
import numpy as np

from ecg_arrhythmia import config
from ecg_arrhythmia.data import aggregate_superclasses, binary_label
from ecg_arrhythmia.features import extract_record
from ecg_arrhythmia.preprocess import preprocess_record

CODE_MAP = {"NORM": "NORM", "IMI": "MI", "ASMI": "MI", "NDT": "STTC", "LVH": "HYP"}


def test_aggregate_pure_norm():
    assert aggregate_superclasses({"NORM": 100.0}, CODE_MAP) == ["NORM"]


def test_aggregate_dedupes_superclass():
    # two MI codes -> single MI superclass
    assert aggregate_superclasses({"IMI": 100.0, "ASMI": 80.0}, CODE_MAP) == ["MI"]


def test_aggregate_multilabel_sorted():
    out = aggregate_superclasses({"NORM": 100.0, "NDT": 50.0}, CODE_MAP)
    assert out == ["NORM", "STTC"]


def test_aggregate_ignores_unknown_codes():
    assert aggregate_superclasses({"SR": 0.0}, CODE_MAP) == []


def test_binary_label():
    assert binary_label(["NORM"]) == 0
    assert binary_label(["NORM", "MI"]) == 1
    assert binary_label(["MI"]) == 1
    assert binary_label([]) == 1


def test_preprocess_shape_and_finiteness():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((config.SIGNAL_LEN, config.N_LEADS)).astype(np.float32)
    proc = preprocess_record(raw)
    assert proc.shape == (config.SIGNAL_LEN, config.N_LEADS)
    assert np.isfinite(proc).all()


def test_extract_record_no_nans():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((config.SIGNAL_LEN, config.N_LEADS)).astype(np.float32)
    feats = extract_record(raw)
    assert len(feats) > 0
    assert all(np.isfinite(v) for v in feats.values())
