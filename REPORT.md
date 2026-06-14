# Automated Arrhythmia Screening from 12-Lead ECG Using the PTB-XL Dataset

**Project Report** · *Languages: English · [ไทย](report_th.md) · [Tiếng Việt](report_vn.md)*
**Dataset:** PTB-XL (12-lead clinical ECG)
**Analysis type:** Biomedical signal processing with classical machine-learning
classification (multi-label diagnostic superclass and binary normal-vs-abnormal)

> **Academic Disclaimer:** This project is for educational purposes only. All
> findings are based on a public research dataset and must not be applied in any
> clinical or diagnostic context.

---

## 1. Introduction

Cardiovascular disease is the leading cause of death worldwide, and the
electrocardiogram (ECG) is the most widely used, low-cost, non-invasive tool for
assessing cardiac electrical activity. A standard 12-lead ECG views the heart
from different angles, letting clinicians detect rhythm disturbances, conduction
abnormalities, ischemia, and structural changes such as hypertrophy.

Manual interpretation is time-consuming and needs expertise that is not always
available at the point of care. Automated analysis can provide a fast first-pass
screen, flag abnormal recordings for prioritized review, and support triage. The
clinical value of such a system depends less on raw accuracy than on
**sensitivity**: missing a genuinely abnormal ECG is far more costly than a
false alarm.

This project builds an end-to-end pipeline that classifies 12-lead ECGs using
**classical biomedical signal processing** (filtering, R-peak detection,
spectral analysis) with conventional machine-learning models, without deep
learning. The aim is to quantify how much diagnostic signal interpretable
hand-crafted features recover, and to characterize where the approach succeeds
and where it reaches its limits.

---

## 2. Dataset Description

### 2.1 Overview

**PTB-XL** (Wagner et al., 2020) is one of the largest public clinical ECG
collections: **21,799 ten-second 12-lead recordings** from **18,869 patients**,
each annotated by cardiologists with the SCP-ECG vocabulary. This project uses
the **100 Hz** records, standard for machine-learning work on PTB-XL.

### 2.2 Signal Format

| Property | Value |
|----------|-------|
| Leads | 12 (I, II, III, aVR, aVL, aVF, V1-V6) |
| Duration | 10 seconds |
| Sampling rate | 100 Hz (1,000 samples per lead) |
| Storage | WFDB format (`.dat` signal + `.hea` header) |

### 2.3 Diagnostic Superclasses

Each record's SCP codes map to one or more of five **superclasses**, so the
primary task is **multi-label**.

| Code | Meaning |
|------|---------|
| NORM | Normal ECG |
| MI | Myocardial infarction |
| STTC | ST/T-wave change |
| CD | Conduction disturbance |
| HYP | Hypertrophy |

### 2.4 Class Distribution

After excluding records with no diagnostic superclass, **21,388**
recordings remain. The data is imbalanced; counts exceed the record total
because labels are multi-label.

| Superclass | Count | Prevalence |
|-----------|-------|-----------|
| NORM (Normal ECG) | 9,514 | 44.5% |
| MI (Myocardial infarction) | 5,469 | 25.6% |
| STTC (ST/T change) | 5,235 | 24.5% |
| CD (Conduction disturbance) | 4,898 | 22.9% |
| HYP (Hypertrophy) | 2,649 | 12.4% |

A record is **normal** only if its sole superclass is NORM, giving
9,069 normal and 12,319 abnormal records
(57.6% abnormal).

![Figure 1: Class prevalence](reports/figures/eda_class_prevalence.png)
*Figure 1: Diagnostic-superclass prevalence (left) and binary class balance (right).*

This imbalance is the single most important modelling consideration: it drives
the class weighting, the metrics, and the per-class performance gaps below.

---

## 3. Research Questions

**Primary question.** How much diagnostic information about 12-lead ECGs can be
recovered using only classical, interpretable signal-processing features?

**Sub-questions.**
1. Can hand-crafted features separate the five superclasses, and which are
   easiest or hardest?
2. How well can a metadata-free model screen normal from abnormal ECGs?
3. What is the sensitivity-vs-specificity trade-off for screening?
4. Which signal features carry the most diagnostic weight, and do they align
   with known cardiac physiology?

---

## 4. Methodology

### 4.1 Preprocessing
Each record is filtered with a **4th-order Butterworth bandpass (0.5-40 Hz,
zero-phase `filtfilt`)** to remove baseline wander and high-frequency noise,
then **z-score normalized per lead**.

![Figure 2: Preprocessed ECG](reports/figures/ecg_sample.png)
*Figure 2: A single preprocessed record, all 12 leads, with detected R-peaks
(red x) on lead II.*

### 4.2 Feature Extraction
A fixed-length **128-feature** vector per record: **rhythm/HRV** from lead-II
R-peaks; **per-lead time-domain statistics** (std, RMS, range, skewness,
kurtosis, zero-crossing rate); and **per-lead spectral band powers** (Welch's
method in four bands).

### 4.3 Data Split
The official PTB-XL `strat_fold` partition prevents patient leakage:
**folds 1-8 train (17,084)**, **fold 9 validation (2,146)**,
**fold 10 test (2,158)**. All results below are on fold 10.

### 4.4 Models
Logistic Regression, Random Forest (300 trees), and Histogram Gradient Boosting
in a `StandardScaler` pipeline, each with `class_weight="balanced"`. The best
model on validation is kept.

### 4.5 Evaluation Metrics
**ROC-AUC** (threshold-independent ranking) is the headline metric; F1,
accuracy, average precision, sensitivity and specificity support it.

---

## 5. Results

The best model for both tasks was **hist_gbdt**.

| Task | Headline metric |
|------|-----------------|
| 5-superclass (multi-label) | macro-AUC **0.859**, micro-AUC 0.884 |
| Binary (normal vs abnormal) | ROC-AUC **0.894**, accuracy 0.811, F1 0.830 |

### 5.1 Multi-Label Discrimination

![Figure 3: Per-superclass ROC](reports/figures/multilabel_roc.png)
*Figure 3: Per-superclass ROC curves (macro-AUC 0.859).*

All five ROC curves sit well above chance, confirming real diagnostic signal in
hand-crafted features. **NORM** is easiest
(AUC 0.901) and **HYP** hardest
(AUC 0.797).

| Superclass | ROC-AUC | F1 | Accuracy |
|-----------|---------|----|----------|
| NORM | 0.901 | 0.805 | 0.819 |
| STTC | 0.888 | 0.627 | 0.842 |
| CD | 0.871 | 0.657 | 0.866 |
| MI | 0.837 | 0.540 | 0.804 |
| HYP | 0.797 | 0.248 | 0.885 |

![Figure 4: Per-class AUC vs F1](reports/figures/multilabel_per_class.png)
*Figure 4: Per-superclass ROC-AUC beside F1 at the 0.5 threshold.*

AUC stays high while **F1 falls for rare classes**, the imbalance gap. Per-label
(Hamming) accuracy averages 84.3%, but getting all five labels
exactly right happens only 50.4% of the time, which is why AUC,
not accuracy, is the fair headline.

### 5.2 Binary Screening

![Figure 5: Binary confusion matrix](reports/figures/binary_confusion.png)
*Figure 5: Normal-vs-abnormal confusion matrix at threshold 0.5
(sensitivity 80.0%, specificity 82.6%).*

The costly error is the lower-left cell: **249 false negatives**,
abnormal ECGs called normal. The current point gives sensitivity
80.0% against specificity 82.6%.

![Figure 6: Binary ROC and precision-recall](reports/figures/binary_roc_pr.png)
*Figure 6: ROC (AUC 0.894) and precision-recall
(average precision 0.929).*

The precision-recall curve is the more honest view under imbalance and sits well
above the 0.58 positive baseline.

### 5.3 Feature Importance

![Figure 7: Feature importance](reports/figures/feature_importance.png)
*Figure 7: Top-20 features by importance.*

The most informative features are dominated by:

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

These map to recognizable physiology: per-lead shape statistics (skewness,
zero-crossing rate) reflect QRS/T morphology, band powers capture frequency
content, and HRV terms capture rhythm.

---

## 6. Discussion

- **Diagnostic signal is real but uneven.** macro-AUC 0.859
  from features alone is a strong classical baseline; performance tracks how
  localized each pattern is (NORM easiest, HYP hardest).
- **Imbalance shapes everything.** NORM dominates, so accuracy misleads and rare
  classes show an AUC-vs-F1 gap, motivating per-class threshold tuning.
- **Metadata-free screening is viable.** Binary AUC 0.894 with
  sensitivity 80.0% is a usable triage signal, not a diagnosis.
- **Features are physiologically sensible**, making the model auditable.

---

## 7. Limitations

1. Classical features cap below deep learning (~0.93 macro-AUC for published
   CNNs vs 0.859 here).
2. R-peaks are detected on lead II only.
3. Reported F1/accuracy use a fixed 0.5 threshold.
4. Only 100 Hz signals were used.
5. SCP-code likelihoods are collapsed to binary superclass presence.
6. Academic use only; not clinically validated.

---

## 8. Conclusion

Classical signal processing recovers meaningful structure from 12-lead ECGs:
**macro-AUC 0.859** across five superclasses and **ROC-AUC
0.894** for binary screening on a patient-disjoint test fold,
with physiologically interpretable features. It cannot replace expert reading or
deep models, but it is a transparent baseline for triage tooling. Future work:
per-class threshold tuning, wavelet/beat-template features, 500 Hz signals, and
a 1-D CNN baseline.

---

## References

Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation, 101*(23), e215-e220.

Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
*IEEE Transactions on Biomedical Engineering, BME-32*(3), 230-236.

Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning
for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE JBHI, 25*(5),
1519-1528.

Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography
dataset. *Scientific Data, 7*, 154.
