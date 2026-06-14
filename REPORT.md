# ECG Arrhythmia Analyzer: Results Report

Classical signal-processing + machine-learning pipeline over **PTB-XL**
(12-lead ECG, 100 Hz). Models are trained on strat-folds 1-8, selected on
fold 9, and all numbers below are on the **held-out test fold 10**
(2158 records).

| Task | Best model | Headline metric |
|------|-----------|-----------------|
| 5-superclass (multi-label) | `hist_gbdt` | macro-AUC **0.859** (micro 0.884) |
| Binary normal vs abnormal | `hist_gbdt` | ROC-AUC **0.894**, acc 0.811, F1 0.830 |

---

## 1. Dataset composition: `eda_class_prevalence.png`

![class prevalence](figures/eda_class_prevalence.png)

The left panel shows diagnostic-superclass prevalence (NORM 44%, MI 26%, STTC 24%, CD 23%, HYP 12%);
because PTB-XL is multi-label these sum past 100%. The right panel shows the
binary split (58% abnormal).

**Analysis.** The dataset is **imbalanced**: NORM dominates while HYP is rare
(~12%). This directly predicts where the models
struggle: minority classes have fewer positive examples to learn from, and that
is why every classifier below uses `class_weight="balanced"` and why we report
AUC/F1 rather than raw accuracy.

## 2. Signal quality: `ecg_sample.png`

![sample ECG](figures/ecg_sample.png)

A single preprocessed record, all 12 leads, with detected R-peaks (red ×) on
lead II.

**Analysis.** This validates the front of the pipeline: the 0.5-40 Hz
Butterworth bandpass has removed baseline wander (traces are centred) and the
R-peak detector fires on QRS complexes, not noise. Those peaks drive the
HRV/rhythm features, so a clean detection here is the prerequisite for the
rhythm-related features being meaningful.

## 3. Multi-label discrimination: `multilabel_roc.png`

![multilabel ROC](figures/multilabel_roc.png)

Per-superclass ROC curves; macro-AUC **0.859**.

**Analysis.** All five curves sit well above the diagonal, so hand-crafted
features carry real diagnostic signal. **NORM** is easiest
(AUC 0.901) and **HYP** hardest
(AUC 0.797). The harder classes are the ones whose
signature is subtle or spread across leads, where a fixed feature set loses to
morphology a CNN would learn end-to-end.

## 4. Per-class AUC vs F1: `multilabel_per_class.png`

![per class](figures/multilabel_per_class.png)

AUC (ranking quality) beside F1 (quality at the 0.5 threshold) per class.

**Analysis.** AUC stays high while **F1 drops for rare classes**, the
classic imbalance gap. A model can rank HYP positives correctly
(good AUC) yet still miss them at a fixed 0.5 cut (low F1). Practically this
says the operating threshold should be tuned per class rather than left at 0.5.

## 5. Binary confusion matrix: `binary_confusion.png`

![confusion](figures/binary_confusion.png)

Normal-vs-abnormal at threshold 0.5. Sensitivity **80.0%**,
specificity **82.6%**.

**Analysis.** For a screening tool the costly error is the bottom-left cell
(249 false negatives, abnormal ECGs called normal). Sensitivity
80.0% vs specificity 82.6% shows the current trade-off; lowering the
threshold would catch more abnormals at the cost of more false alarms (see the
PR curve next).

## 6. Binary ROC & precision-recall: `binary_roc_pr.png`

![roc pr](figures/binary_roc_pr.png)

ROC-AUC **0.894**, average precision **0.929**.

**Analysis.** ROC summarizes threshold-independent separability; the PR curve
is the more honest view under imbalance and sits well above the
0.58 baseline. Together they let you pick an operating
point for a target sensitivity instead of accepting the default threshold.

## 7. What the model keys on: `feature_importance.png`

![importance](figures/feature_importance.png)

Top-20 features by importance.

**Analysis.** The most informative features are dominated by:

1. `II_skew`
2. `aVR_zcr`
3. `V4_zcr`
4. `V6_bp_vlf`
5. `V2_bp_lf`
6. `I_bp_hf`
7. `V1_skew`
8. `V5_zcr`
9. `aVF_skew`
10. `V1_bp_mf`

These tie back to recognizable physiology: per-lead shape statistics
(skew, zero-crossing rate) reflect QRS/T morphology, the spectral band powers
capture waveform frequency content, and the HRV terms capture rhythm. That
interpretability is the upside of the classical approach.

---

## Takeaways & next steps

* Hand-crafted features reach macro-AUC **0.859** (multi-label)
  and **0.894** AUC (binary): solid, interpretable baselines.
* The ceiling is set by class imbalance and by morphology that fixed features
  miss; published 1-D CNNs reach ~0.93 macro-AUC.
* Next: per-class threshold tuning, wavelet/template features, and a CNN
  baseline for comparison.
