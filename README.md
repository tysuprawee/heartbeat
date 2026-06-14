# ECG Arrhythmia Detection System

Classical biomedical **signal-processing + machine-learning** pipeline that
classifies 12-lead ECGs from the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
dataset. Two tasks:

- **5-superclass multi-label**: NORM, MI, STTC, CD, HYP (the official PTB-XL
  diagnostic benchmark)
- **Binary**: normal vs. abnormal

**Stack:** Python · NumPy · SciPy · scikit-learn · Matplotlib (+ pandas, wfdb)

## Results (held-out test fold 10)

| Task | Best model | Headline metric |
|------|-----------|-----------------|
| 5-superclass (multi-label) | HistGBDT | **macro-AUC 0.859** (micro 0.884) |
| Binary normal vs abnormal  | HistGBDT | **ROC-AUC 0.894**, acc 0.811, F1 0.830 |

Per-class test AUC: NORM 0.901 · STTC 0.888 · CD 0.871 · MI 0.837 · HYP 0.797.

See the full report for every figure with a per-graph analysis:
**[English](REPORT.md)** · **[ไทย](report_th.md)** · **[Tiếng Việt](report_vn.md)**.

## How it works

```
raw 12-lead ECG (100 Hz, 10 s)
  │  preprocess.py   0.5-40 Hz Butterworth bandpass (filtfilt) + per-lead z-score
  ▼
  │  features.py     R-peak detection (lead II) → HRV; per-lead time stats
  │                  (std/RMS/range/skew/kurtosis/ZCR); Welch spectral band power
  ▼  128 features/record
  │  train.py        StandardScaler → LogReg / RandomForest / HistGBDT
  │                  fit on folds 1-8, select best on fold 9
  ▼
  │  evaluate.py     score on fold 10 → metrics + figures + REPORT.md
```

The official `strat_fold` split (folds 1-8 train / 9 val / 10 test) is used, so
there is no patient leakage between splits.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download PTB-XL into the project root (≈3 GB; only the 100 Hz `records100/` is
required):

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

## Run the pipeline

```bash
python -m ecg_arrhythmia.data --build-cache     # signals + labels cache  (~40 s)
python -m ecg_arrhythmia.features               # 128-feature matrix      (~2.5 min)
python -m ecg_arrhythmia.train --task both      # train + select models   (~40 s)
python -m ecg_arrhythmia.evaluate --task both   # metrics, figures, REPORT.md
python -m ecg_arrhythmia.viz --ecg-id 1         # plot one 12-lead record
```

Quick end-to-end smoke test on a small subset (no cache written):

```bash
python -m ecg_arrhythmia.train --task both --sample 500
```

## Project layout

```
ecg_arrhythmia/
  config.py       paths, signal constants, label space, fold split
  data.py         metadata, superclass aggregation, signal cache
  preprocess.py   SciPy bandpass + normalization
  features.py     R-peak/HRV + per-lead time + spectral features
  train.py        scikit-learn pipelines for both tasks
  evaluate.py     test-fold metrics, Matplotlib figures, REPORT.md
  viz.py          12-lead ECG plot with R-peaks
tests/            pytest unit tests (labels, preprocessing, features)
reports/figures/  generated PNGs
```

## Tests

```bash
pytest -q
```

## Notes & limitations

- Uses the 100 Hz records (standard for PTB-XL ML work); the 500 Hz set is not
  needed.
- Hand-crafted features are interpretable but cap below end-to-end deep models
  (published 1-D CNNs reach ~0.93 macro-AUC). Documented next steps: per-class
  threshold tuning, wavelet/template features, and a CNN baseline.
- Not a medical device; research/educational use only.
