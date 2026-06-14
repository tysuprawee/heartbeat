"""Multi-language report renderers (English / Thai / Vietnamese).

Each ``render_*`` function takes a context dict of computed metrics (built in
``evaluate.write_report``) and returns a complete Markdown report. Keeping the
numbers in one context dict means all language versions stay in sync whenever
the pipeline is re-run.
"""
from __future__ import annotations

from . import config

F = "reports/figures"  # image paths relative to the report files at repo root
SC = config.SUPERCLASSES


def _by_count(ctx):
    return sorted(SC, key=lambda s: -ctx["counts"][s])


def _by_auc(ctx):
    return sorted(SC, key=lambda s: -ctx["per_auc"][s])


def _feat_lines(ctx):
    return "\n".join(f"{i+1}. `{f}`" for i, f in enumerate(ctx["top_feats"][:10]))


# ---------------------------------------------------------------- English ---
def render_en(ctx) -> str:
    names = {"NORM": "Normal ECG", "MI": "Myocardial infarction",
             "STTC": "ST/T change", "CD": "Conduction disturbance",
             "HYP": "Hypertrophy"}
    dist = "\n".join(f"| {s} ({names[s]}) | {ctx['counts'][s]:,} | {ctx['prevalence'][s]:.1%} |"
                     for s in _by_count(ctx))
    mlr = "\n".join(f"| {s} | {ctx['per_auc'][s]:.3f} | {ctx['per_f1'][s]:.3f} | {ctx['per_acc'][s]:.3f} |"
                    for s in _by_auc(ctx))
    return f"""# Automated Arrhythmia Screening from 12-Lead ECG Using the PTB-XL Dataset

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

After excluding records with no diagnostic superclass, **{ctx['n_total']:,}**
recordings remain. The data is imbalanced; counts exceed the record total
because labels are multi-label.

| Superclass | Count | Prevalence |
|-----------|-------|-----------|
{dist}

A record is **normal** only if its sole superclass is NORM, giving
{ctx['n_norm']:,} normal and {ctx['n_abn']:,} abnormal records
({ctx['binary_abnormal']:.1%} abnormal).

![Figure 1: Class prevalence]({F}/eda_class_prevalence.png)
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

![Figure 2: Preprocessed ECG]({F}/ecg_sample.png)
*Figure 2: A single preprocessed record, all 12 leads, with detected R-peaks
(red x) on lead II.*

### 4.2 Feature Extraction
A fixed-length **128-feature** vector per record: **rhythm/HRV** from lead-II
R-peaks; **per-lead time-domain statistics** (std, RMS, range, skewness,
kurtosis, zero-crossing rate); and **per-lead spectral band powers** (Welch's
method in four bands).

### 4.3 Data Split
The official PTB-XL `strat_fold` partition prevents patient leakage:
**folds 1-8 train ({ctx['n_tr']:,})**, **fold 9 validation ({ctx['n_va']:,})**,
**fold 10 test ({ctx['n_te']:,})**. All results below are on fold 10.

### 4.4 Models
Logistic Regression, Random Forest (300 trees), and Histogram Gradient Boosting
in a `StandardScaler` pipeline, each with `class_weight="balanced"`. The best
model on validation is kept.

### 4.5 Evaluation Metrics
**ROC-AUC** (threshold-independent ranking) is the headline metric; F1,
accuracy, average precision, sensitivity and specificity support it.

---

## 5. Results

The best model for both tasks was **{ctx['model_ml']}**.

| Task | Headline metric |
|------|-----------------|
| 5-superclass (multi-label) | macro-AUC **{ctx['macro_auc']:.3f}**, micro-AUC {ctx['micro_auc']:.3f} |
| Binary (normal vs abnormal) | ROC-AUC **{ctx['bi_auc']:.3f}**, accuracy {ctx['bi_acc']:.3f}, F1 {ctx['bi_f1']:.3f} |

### 5.1 Multi-Label Discrimination

![Figure 3: Per-superclass ROC]({F}/multilabel_roc.png)
*Figure 3: Per-superclass ROC curves (macro-AUC {ctx['macro_auc']:.3f}).*

All five ROC curves sit well above chance, confirming real diagnostic signal in
hand-crafted features. **{ctx['best_sc']}** is easiest
(AUC {ctx['per_auc'][ctx['best_sc']]:.3f}) and **{ctx['worst_sc']}** hardest
(AUC {ctx['per_auc'][ctx['worst_sc']]:.3f}).

| Superclass | ROC-AUC | F1 | Accuracy |
|-----------|---------|----|----------|
{mlr}

![Figure 4: Per-class AUC vs F1]({F}/multilabel_per_class.png)
*Figure 4: Per-superclass ROC-AUC beside F1 at the 0.5 threshold.*

AUC stays high while **F1 falls for rare classes**, the imbalance gap. Per-label
(Hamming) accuracy averages {ctx['hamming_acc']:.1%}, but getting all five labels
exactly right happens only {ctx['subset_acc']:.1%} of the time, which is why AUC,
not accuracy, is the fair headline.

### 5.2 Binary Screening

![Figure 5: Binary confusion matrix]({F}/binary_confusion.png)
*Figure 5: Normal-vs-abnormal confusion matrix at threshold 0.5
(sensitivity {ctx['sens']:.1%}, specificity {ctx['spec']:.1%}).*

The costly error is the lower-left cell: **{ctx['fn']} false negatives**,
abnormal ECGs called normal. The current point gives sensitivity
{ctx['sens']:.1%} against specificity {ctx['spec']:.1%}.

![Figure 6: Binary ROC and precision-recall]({F}/binary_roc_pr.png)
*Figure 6: ROC (AUC {ctx['bi_auc']:.3f}) and precision-recall
(average precision {ctx['bi_ap']:.3f}).*

The precision-recall curve is the more honest view under imbalance and sits well
above the {ctx['binary_abnormal']:.2f} positive baseline.

### 5.3 Feature Importance

![Figure 7: Feature importance]({F}/feature_importance.png)
*Figure 7: Top-20 features by importance.*

The most informative features are dominated by:

{_feat_lines(ctx)}

These map to recognizable physiology: per-lead shape statistics (skewness,
zero-crossing rate) reflect QRS/T morphology, band powers capture frequency
content, and HRV terms capture rhythm.

---

## 6. Discussion

- **Diagnostic signal is real but uneven.** macro-AUC {ctx['macro_auc']:.3f}
  from features alone is a strong classical baseline; performance tracks how
  localized each pattern is ({ctx['best_sc']} easiest, {ctx['worst_sc']} hardest).
- **Imbalance shapes everything.** NORM dominates, so accuracy misleads and rare
  classes show an AUC-vs-F1 gap, motivating per-class threshold tuning.
- **Metadata-free screening is viable.** Binary AUC {ctx['bi_auc']:.3f} with
  sensitivity {ctx['sens']:.1%} is a usable triage signal, not a diagnosis.
- **Features are physiologically sensible**, making the model auditable.

---

## 7. Limitations

1. Classical features cap below deep learning (~0.93 macro-AUC for published
   CNNs vs {ctx['macro_auc']:.3f} here).
2. R-peaks are detected on lead II only.
3. Reported F1/accuracy use a fixed 0.5 threshold.
4. Only 100 Hz signals were used.
5. SCP-code likelihoods are collapsed to binary superclass presence.
6. Academic use only; not clinically validated.

---

## 8. Conclusion

Classical signal processing recovers meaningful structure from 12-lead ECGs:
**macro-AUC {ctx['macro_auc']:.3f}** across five superclasses and **ROC-AUC
{ctx['bi_auc']:.3f}** for binary screening on a patient-disjoint test fold,
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
"""


# ------------------------------------------------------------------- Thai ---
def render_th(ctx) -> str:
    names = {"NORM": "ECG ปกติ", "MI": "กล้ามเนื้อหัวใจตาย",
             "STTC": "การเปลี่ยนแปลง ST/T", "CD": "ความผิดปกติของการนำไฟฟ้า",
             "HYP": "ภาวะหัวใจโต"}
    dist = "\n".join(f"| {s} ({names[s]}) | {ctx['counts'][s]:,} | {ctx['prevalence'][s]:.1%} |"
                     for s in _by_count(ctx))
    mlr = "\n".join(f"| {s} | {ctx['per_auc'][s]:.3f} | {ctx['per_f1'][s]:.3f} | {ctx['per_acc'][s]:.3f} |"
                    for s in _by_auc(ctx))
    return f"""# การคัดกรองภาวะหัวใจเต้นผิดจังหวะอัตโนมัติจาก ECG 12 ลีด ด้วยชุดข้อมูล PTB-XL

**รายงานโปรเจกต์** · *ภาษา: [English](REPORT.md) · ไทย · [Tiếng Việt](report_vn.md)*
**ชุดข้อมูล:** PTB-XL (ECG ทางคลินิกแบบ 12 ลีด)
**ประเภทการวิเคราะห์:** การประมวลผลสัญญาณชีวการแพทย์ร่วมกับการจำแนกด้วยการเรียนรู้
ของเครื่องแบบดั้งเดิม (จำแนกกลุ่มวินิจฉัยแบบหลายป้าย และจำแนกแบบสองกลุ่ม ปกติ/ผิดปกติ)

> **ข้อจำกัดความรับผิดชอบเชิงวิชาการ:** โปรเจกต์นี้จัดทำเพื่อการศึกษาเท่านั้น ผลลัพธ์
> ทั้งหมดอ้างอิงจากชุดข้อมูลวิจัยสาธารณะ และต้องไม่นำไปใช้ในบริบททางคลินิกหรือการวินิจฉัยใด ๆ

---

## 1. บทนำ

โรคหัวใจและหลอดเลือดเป็นสาเหตุการเสียชีวิตอันดับหนึ่งของโลก และคลื่นไฟฟ้าหัวใจ (ECG)
คือเครื่องมือที่ใช้กันแพร่หลายที่สุด มีต้นทุนต่ำ และไม่รุกล้ำร่างกาย สำหรับประเมินกิจกรรม
ทางไฟฟ้าของหัวใจ ECG มาตรฐานแบบ 12 ลีดมองหัวใจจากหลายมุม ทำให้แพทย์ตรวจพบความผิดปกติ
ของจังหวะ การนำไฟฟ้า ภาวะขาดเลือด และการเปลี่ยนแปลงเชิงโครงสร้าง เช่น ภาวะหัวใจโต ได้

การอ่านด้วยมือใช้เวลานานและต้องอาศัยความเชี่ยวชาญที่ไม่ได้มีอยู่เสมอ ณ จุดดูแลผู้ป่วย
การวิเคราะห์อัตโนมัติช่วยคัดกรองเบื้องต้นได้อย่างรวดเร็ว ทำเครื่องหมายสัญญาณที่ผิดปกติ
เพื่อให้ตรวจทานก่อน และสนับสนุนการคัดแยกผู้ป่วย คุณค่าทางคลินิกของระบบเช่นนี้ขึ้นกับ
**ความไว (sensitivity)** มากกว่าความแม่นยำดิบ เพราะการพลาด ECG ที่ผิดปกติจริงมีต้นทุน
สูงกว่าการแจ้งเตือนผิดพลาดมาก

โปรเจกต์นี้สร้างไปป์ไลน์แบบครบวงจรที่จำแนก ECG 12 ลีด โดยใช้**การประมวลผลสัญญาณชีว
การแพทย์แบบดั้งเดิม** (การกรองสัญญาณ การตรวจจับจุดยอด R และการวิเคราะห์เชิงสเปกตรัม)
ร่วมกับโมเดลการเรียนรู้ของเครื่องทั่วไป โดยไม่ใช้การเรียนรู้เชิงลึก เป้าหมายคือวัดว่า
ฟีเจอร์ที่ออกแบบด้วยมือและตีความได้ สามารถดึงสัญญาณวินิจฉัยออกมาได้มากเพียงใด

---

## 2. รายละเอียดชุดข้อมูล

### 2.1 ภาพรวม

**PTB-XL** (Wagner และคณะ, 2020) เป็นหนึ่งในชุดข้อมูล ECG ทางคลินิกสาธารณะที่ใหญ่ที่สุด
ประกอบด้วยการบันทึก 12 ลีด ความยาว 10 วินาที จำนวน **21,799 รายการ** จากผู้ป่วย
**18,869 คน** แต่ละรายการมีการกำกับโดยแพทย์โรคหัวใจด้วยรหัส SCP-ECG โปรเจกต์นี้ใช้
ข้อมูลที่ **100 Hz** ซึ่งเป็นมาตรฐานสำหรับงานการเรียนรู้ของเครื่องบน PTB-XL

### 2.2 รูปแบบสัญญาณ

| คุณสมบัติ | ค่า |
|----------|-----|
| ลีด | 12 (I, II, III, aVR, aVL, aVF, V1-V6) |
| ความยาว | 10 วินาที |
| อัตราการสุ่ม | 100 Hz (1,000 จุดต่อลีด) |
| การจัดเก็บ | รูปแบบ WFDB (สัญญาณ `.dat` + ส่วนหัว `.hea`) |

### 2.3 กลุ่มวินิจฉัยหลัก (Superclass)

รหัส SCP ของแต่ละรายการจับคู่กับกลุ่มวินิจฉัยหลักหนึ่งกลุ่มหรือมากกว่า จากทั้งหมดห้ากลุ่ม
งานหลักจึงเป็นการจำแนก **แบบหลายป้าย (multi-label)**

| รหัส | ความหมาย |
|------|----------|
| NORM | ECG ปกติ |
| MI | กล้ามเนื้อหัวใจตาย |
| STTC | การเปลี่ยนแปลงคลื่น ST/T |
| CD | ความผิดปกติของการนำไฟฟ้า |
| HYP | ภาวะหัวใจโต |

### 2.4 การกระจายของกลุ่ม

หลังจากตัดรายการที่ไม่มีกลุ่มวินิจฉัยออก เหลือ **{ctx['n_total']:,}** รายการ ข้อมูลมีความ
ไม่สมดุล ผลรวมของจำนวนเกินจำนวนรายการทั้งหมดเพราะเป็นป้ายแบบหลายป้าย

| กลุ่มวินิจฉัย | จำนวน | สัดส่วน |
|-------------|-------|---------|
{dist}

รายการจะเป็น **ปกติ** ก็ต่อเมื่อมีกลุ่มวินิจฉัยเดียวคือ NORM เท่านั้น ทำให้มีรายการปกติ
{ctx['n_norm']:,} รายการ และผิดปกติ {ctx['n_abn']:,} รายการ
(ผิดปกติ {ctx['binary_abnormal']:.1%})

![ภาพที่ 1: สัดส่วนของกลุ่ม]({F}/eda_class_prevalence.png)
*ภาพที่ 1: สัดส่วนกลุ่มวินิจฉัยหลัก (ซ้าย) และสมดุลของสองกลุ่ม (ขวา)*

ความไม่สมดุลนี้เป็นปัจจัยสำคัญที่สุดในการสร้างโมเดล โดยกำหนดทั้งการถ่วงน้ำหนักกลุ่ม
ตัวชี้วัด และช่องว่างของประสิทธิภาพรายกลุ่มด้านล่าง

---

## 3. คำถามวิจัย

**คำถามหลัก** ฟีเจอร์การประมวลผลสัญญาณแบบดั้งเดิมที่ตีความได้ สามารถดึงข้อมูลวินิจฉัย
จาก ECG 12 ลีดออกมาได้มากเพียงใด

**คำถามย่อย**
1. ฟีเจอร์ที่ออกแบบด้วยมือแยกกลุ่มวินิจฉัยทั้งห้าได้หรือไม่ และกลุ่มใดง่ายหรือยากที่สุด
2. โมเดลที่ไม่ใช้ข้อมูลอภิพันธุ์ คัดกรองปกติ/ผิดปกติได้ดีเพียงใด
3. การแลกเปลี่ยนระหว่างความไวและความจำเพาะสำหรับการคัดกรองเป็นอย่างไร
4. ฟีเจอร์สัญญาณใดมีน้ำหนักวินิจฉัยมากที่สุด และสอดคล้องกับสรีรวิทยาหัวใจหรือไม่

---

## 4. ระเบียบวิธี

### 4.1 การประมวลผลเบื้องต้น
แต่ละรายการถูกกรองด้วย **ตัวกรองแบนด์พาส Butterworth อันดับ 4 (0.5-40 Hz แบบเฟสศูนย์
`filtfilt`)** เพื่อกำจัดการเลื่อนของเส้นฐานและสัญญาณรบกวนความถี่สูง จากนั้น **ปรับมาตรฐาน
แบบ z-score รายลีด**

![ภาพที่ 2: ECG หลังประมวลผล]({F}/ecg_sample.png)
*ภาพที่ 2: รายการเดียวหลังประมวลผล ครบทั้ง 12 ลีด พร้อมจุดยอด R ที่ตรวจพบ
(เครื่องหมาย x สีแดง) บนลีด II*

### 4.2 การสกัดฟีเจอร์
เวกเตอร์ฟีเจอร์ความยาวคงที่ **128 ฟีเจอร์** ต่อรายการ ได้แก่ **จังหวะ/HRV** จากจุดยอด R
บนลีด II; **สถิติเชิงเวลารายลีด** (ส่วนเบี่ยงเบนมาตรฐาน, RMS, พิสัย, ความเบ้, ความโด่ง,
อัตราการตัดศูนย์); และ **กำลังแถบความถี่รายลีด** (วิธี Welch ในสี่แถบ)

### 4.3 การแบ่งข้อมูล
การแบ่ง `strat_fold` อย่างเป็นทางการของ PTB-XL ป้องกันการรั่วของข้อมูลผู้ป่วย:
**โฟลด์ 1-8 ฝึก ({ctx['n_tr']:,})**, **โฟลด์ 9 ตรวจสอบ ({ctx['n_va']:,})**,
**โฟลด์ 10 ทดสอบ ({ctx['n_te']:,})** ผลทั้งหมดด้านล่างมาจากโฟลด์ 10

### 4.4 โมเดล
Logistic Regression, Random Forest (300 ต้น) และ Histogram Gradient Boosting
ในไปป์ไลน์ `StandardScaler` แต่ละตัวใช้ `class_weight="balanced"` และเลือกโมเดลที่ดีที่สุด
บนชุดตรวจสอบ

### 4.5 ตัวชี้วัด
**ROC-AUC** (การจัดอันดับที่ไม่ขึ้นกับค่าขีดแบ่ง) เป็นตัวชี้วัดหลัก เสริมด้วย F1, ความแม่นยำ,
average precision, ความไว และความจำเพาะ

---

## 5. ผลลัพธ์

โมเดลที่ดีที่สุดของทั้งสองงานคือ **{ctx['model_ml']}**

| งาน | ตัวชี้วัดหลัก |
|-----|-------------|
| 5 กลุ่ม (หลายป้าย) | macro-AUC **{ctx['macro_auc']:.3f}**, micro-AUC {ctx['micro_auc']:.3f} |
| สองกลุ่ม (ปกติ/ผิดปกติ) | ROC-AUC **{ctx['bi_auc']:.3f}**, ความแม่นยำ {ctx['bi_acc']:.3f}, F1 {ctx['bi_f1']:.3f} |

### 5.1 การแยกแยะแบบหลายป้าย

![ภาพที่ 3: ROC รายกลุ่ม]({F}/multilabel_roc.png)
*ภาพที่ 3: เส้น ROC รายกลุ่มวินิจฉัย (macro-AUC {ctx['macro_auc']:.3f})*

เส้น ROC ทั้งห้าอยู่เหนือเส้นสุ่มอย่างชัดเจน ยืนยันว่าฟีเจอร์ที่ออกแบบด้วยมือมีสัญญาณ
วินิจฉัยจริง กลุ่ม **{ctx['best_sc']}** ง่ายที่สุด (AUC {ctx['per_auc'][ctx['best_sc']]:.3f})
และ **{ctx['worst_sc']}** ยากที่สุด (AUC {ctx['per_auc'][ctx['worst_sc']]:.3f})

| กลุ่มวินิจฉัย | ROC-AUC | F1 | ความแม่นยำ |
|-------------|---------|----|-----------|
{mlr}

![ภาพที่ 4: AUC เทียบ F1 รายกลุ่ม]({F}/multilabel_per_class.png)
*ภาพที่ 4: ROC-AUC รายกลุ่ม เทียบกับ F1 ที่ค่าขีดแบ่ง 0.5*

AUC ยังคงสูงในขณะที่ **F1 ลดลงสำหรับกลุ่มที่พบน้อย** อันเป็นช่องว่างจากความไม่สมดุล
ความแม่นยำรายป้าย (Hamming) เฉลี่ย {ctx['hamming_acc']:.1%} แต่การทายถูกครบทั้งห้าป้าย
เกิดขึ้นเพียง {ctx['subset_acc']:.1%} ของเวลา จึงใช้ AUC ไม่ใช่ความแม่นยำ เป็นตัวชี้วัดหลัก

### 5.2 การคัดกรองแบบสองกลุ่ม

![ภาพที่ 5: เมทริกซ์ความสับสน]({F}/binary_confusion.png)
*ภาพที่ 5: เมทริกซ์ความสับสนปกติ/ผิดปกติ ที่ค่าขีดแบ่ง 0.5
(ความไว {ctx['sens']:.1%}, ความจำเพาะ {ctx['spec']:.1%})*

ข้อผิดพลาดที่มีต้นทุนสูงคือช่องล่างซ้าย: **ผลลบลวง {ctx['fn']} ราย** คือ ECG ที่ผิดปกติแต่
ถูกทายว่าปกติ จุดทำงานปัจจุบันให้ความไว {ctx['sens']:.1%} เทียบกับความจำเพาะ {ctx['spec']:.1%}

![ภาพที่ 6: ROC และ precision-recall]({F}/binary_roc_pr.png)
*ภาพที่ 6: ROC (AUC {ctx['bi_auc']:.3f}) และ precision-recall
(average precision {ctx['bi_ap']:.3f})*

เส้น precision-recall เป็นมุมมองที่ตรงไปตรงมากว่าภายใต้ความไม่สมดุล และอยู่สูงกว่าเส้น
ฐานบวกที่ {ctx['binary_abnormal']:.2f} อย่างชัดเจน

### 5.3 ความสำคัญของฟีเจอร์

![ภาพที่ 7: ความสำคัญของฟีเจอร์]({F}/feature_importance.png)
*ภาพที่ 7: ฟีเจอร์ 20 อันดับแรกตามความสำคัญ*

ฟีเจอร์ที่ให้ข้อมูลมากที่สุดมาจาก:

{_feat_lines(ctx)}

ฟีเจอร์เหล่านี้สอดคล้องกับสรีรวิทยาที่รู้จัก: สถิติรูปร่างรายลีด (ความเบ้ อัตราการตัดศูนย์)
สะท้อนสัณฐานของ QRS/T, กำลังแถบความถี่จับเนื้อหาเชิงความถี่ และพจน์ HRV จับจังหวะ

---

## 6. อภิปราย

- **สัญญาณวินิจฉัยมีจริงแต่ไม่สม่ำเสมอ** macro-AUC {ctx['macro_auc']:.3f} จากฟีเจอร์เพียง
  อย่างเดียวเป็นเส้นฐานแบบดั้งเดิมที่แข็งแรง ประสิทธิภาพแปรตามว่ารูปแบบแต่ละกลุ่มเฉพาะที่
  มากเพียงใด ({ctx['best_sc']} ง่ายสุด, {ctx['worst_sc']} ยากสุด)
- **ความไม่สมดุลกำหนดทุกอย่าง** NORM มีจำนวนมาก ความแม่นยำจึงทำให้เข้าใจผิด และกลุ่ม
  ที่พบน้อยมีช่องว่าง AUC กับ F1 จึงควรปรับค่าขีดแบ่งรายกลุ่ม
- **การคัดกรองโดยไม่ใช้ข้อมูลอภิพันธุ์ทำได้จริง** AUC สองกลุ่ม {ctx['bi_auc']:.3f} พร้อม
  ความไว {ctx['sens']:.1%} เป็นสัญญาณคัดแยกที่ใช้ได้ ไม่ใช่การวินิจฉัย
- **ฟีเจอร์มีเหตุผลเชิงสรีรวิทยา** ทำให้โมเดลตรวจสอบย้อนได้

---

## 7. ข้อจำกัด

1. ฟีเจอร์แบบดั้งเดิมมีเพดานต่ำกว่าการเรียนรู้เชิงลึก (CNN ที่เผยแพร่ ~0.93 macro-AUC
   เทียบกับ {ctx['macro_auc']:.3f} ที่นี่)
2. ตรวจจับจุดยอด R บนลีด II เท่านั้น
3. F1/ความแม่นยำที่รายงานใช้ค่าขีดแบ่งคงที่ 0.5
4. ใช้สัญญาณ 100 Hz เท่านั้น
5. ค่าความเป็นไปได้ของรหัส SCP ถูกยุบเป็นการมี/ไม่มีกลุ่มแบบสองค่า
6. ใช้เพื่อการศึกษาเท่านั้น ยังไม่ผ่านการตรวจสอบทางคลินิก

---

## 8. สรุป

การประมวลผลสัญญาณแบบดั้งเดิมดึงโครงสร้างที่มีความหมายจาก ECG 12 ลีดได้:
**macro-AUC {ctx['macro_auc']:.3f}** สำหรับห้ากลุ่มวินิจฉัย และ **ROC-AUC
{ctx['bi_auc']:.3f}** สำหรับการคัดกรองสองกลุ่ม บนโฟลด์ทดสอบที่แยกผู้ป่วยกัน พร้อมฟีเจอร์ที่
ตีความเชิงสรีรวิทยาได้ แม้แทนที่การอ่านของผู้เชี่ยวชาญหรือโมเดลเชิงลึกไม่ได้ แต่เป็นเส้นฐาน
ที่โปร่งใสสำหรับเครื่องมือคัดแยก งานในอนาคต: การปรับค่าขีดแบ่งรายกลุ่ม ฟีเจอร์เวฟเล็ต/
แม่แบบการเต้น สัญญาณ 500 Hz และเส้นฐาน CNN แบบ 1 มิติ

---

## เอกสารอ้างอิง

Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation, 101*(23), e215-e220.

Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
*IEEE Transactions on Biomedical Engineering, BME-32*(3), 230-236.

Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning
for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE JBHI, 25*(5),
1519-1528.

Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography
dataset. *Scientific Data, 7*, 154.
"""


# ------------------------------------------------------------- Vietnamese ---
def render_vi(ctx) -> str:
    names = {"NORM": "ECG bình thường", "MI": "Nhồi máu cơ tim",
             "STTC": "Thay đổi ST/T", "CD": "Rối loạn dẫn truyền",
             "HYP": "Phì đại"}
    dist = "\n".join(f"| {s} ({names[s]}) | {ctx['counts'][s]:,} | {ctx['prevalence'][s]:.1%} |"
                     for s in _by_count(ctx))
    mlr = "\n".join(f"| {s} | {ctx['per_auc'][s]:.3f} | {ctx['per_f1'][s]:.3f} | {ctx['per_acc'][s]:.3f} |"
                    for s in _by_auc(ctx))
    return f"""# Sàng lọc rối loạn nhịp tim tự động từ ECG 12 chuyển đạo với bộ dữ liệu PTB-XL

**Báo cáo dự án** · *Ngôn ngữ: [English](REPORT.md) · [ไทย](report_th.md) · Tiếng Việt*
**Bộ dữ liệu:** PTB-XL (ECG lâm sàng 12 chuyển đạo)
**Loại phân tích:** Xử lý tín hiệu y sinh kết hợp phân loại bằng học máy cổ điển
(phân loại nhóm chẩn đoán đa nhãn và phân loại nhị phân bình thường/bất thường)

> **Tuyên bố miễn trừ học thuật:** Dự án này chỉ phục vụ mục đích giáo dục. Mọi
> kết quả dựa trên bộ dữ liệu nghiên cứu công khai và không được áp dụng trong
> bất kỳ bối cảnh lâm sàng hay chẩn đoán nào.

---

## 1. Giới thiệu

Bệnh tim mạch là nguyên nhân tử vong hàng đầu thế giới, và điện tâm đồ (ECG) là
công cụ phổ biến nhất, chi phí thấp, không xâm lấn để đánh giá hoạt động điện của
tim. ECG 12 chuyển đạo tiêu chuẩn quan sát tim từ nhiều góc, giúp bác sĩ phát hiện
rối loạn nhịp, rối loạn dẫn truyền, thiếu máu cục bộ và các thay đổi cấu trúc như
phì đại.

Đọc thủ công tốn thời gian và đòi hỏi chuyên môn không phải lúc nào cũng có sẵn
tại điểm chăm sóc. Phân tích tự động có thể sàng lọc nhanh ở bước đầu, đánh dấu các
bản ghi bất thường để ưu tiên xem xét, và hỗ trợ phân loại. Giá trị lâm sàng của
hệ thống như vậy phụ thuộc vào **độ nhạy** hơn là độ chính xác thô: bỏ sót một ECG
thực sự bất thường tốn kém hơn nhiều so với một báo động giả.

Dự án này xây dựng một quy trình đầu-cuối phân loại ECG 12 chuyển đạo bằng **xử lý
tín hiệu y sinh cổ điển** (lọc, phát hiện đỉnh R, phân tích phổ) kết hợp các mô
hình học máy thông thường, không dùng học sâu. Mục tiêu là định lượng lượng tín
hiệu chẩn đoán mà các đặc trưng thủ công, dễ diễn giải có thể thu được.

---

## 2. Mô tả bộ dữ liệu

### 2.1 Tổng quan

**PTB-XL** (Wagner và cộng sự, 2020) là một trong những bộ ECG lâm sàng công khai
lớn nhất: **21.799 bản ghi 12 chuyển đạo dài 10 giây** từ **18.869 bệnh nhân**, mỗi
bản ghi được bác sĩ tim mạch chú thích bằng từ vựng SCP-ECG. Dự án dùng dữ liệu
**100 Hz**, tiêu chuẩn cho công việc học máy trên PTB-XL.

### 2.2 Định dạng tín hiệu

| Thuộc tính | Giá trị |
|-----------|---------|
| Chuyển đạo | 12 (I, II, III, aVR, aVL, aVF, V1-V6) |
| Thời lượng | 10 giây |
| Tần số lấy mẫu | 100 Hz (1.000 mẫu mỗi chuyển đạo) |
| Lưu trữ | Định dạng WFDB (tín hiệu `.dat` + tiêu đề `.hea`) |

### 2.3 Nhóm chẩn đoán chính (Superclass)

Mã SCP của mỗi bản ghi ánh xạ tới một hoặc nhiều trong năm **nhóm chính**, nên
nhiệm vụ chính là **đa nhãn (multi-label)**.

| Mã | Ý nghĩa |
|----|---------|
| NORM | ECG bình thường |
| MI | Nhồi máu cơ tim |
| STTC | Thay đổi sóng ST/T |
| CD | Rối loạn dẫn truyền |
| HYP | Phì đại |

### 2.4 Phân bố lớp

Sau khi loại các bản ghi không có nhóm chẩn đoán, còn lại **{ctx['n_total']:,}** bản
ghi. Dữ liệu mất cân bằng; tổng số đếm vượt tổng bản ghi vì nhãn là đa nhãn.

| Nhóm chẩn đoán | Số lượng | Tỷ lệ |
|--------------|---------|-------|
{dist}

Một bản ghi là **bình thường** chỉ khi nhóm duy nhất của nó là NORM, cho
{ctx['n_norm']:,} bản ghi bình thường và {ctx['n_abn']:,} bất thường
({ctx['binary_abnormal']:.1%} bất thường).

![Hình 1: Tỷ lệ các lớp]({F}/eda_class_prevalence.png)
*Hình 1: Tỷ lệ nhóm chẩn đoán chính (trái) và cân bằng nhị phân (phải).*

Sự mất cân bằng này là cân nhắc mô hình hóa quan trọng nhất: nó chi phối trọng số
lớp, các chỉ số, và khoảng cách hiệu năng theo từng lớp bên dưới.

---

## 3. Câu hỏi nghiên cứu

**Câu hỏi chính.** Có thể thu được bao nhiêu thông tin chẩn đoán về ECG 12 chuyển
đạo chỉ bằng các đặc trưng xử lý tín hiệu cổ điển, dễ diễn giải?

**Câu hỏi phụ.**
1. Các đặc trưng thủ công có tách được năm nhóm chính không, và nhóm nào dễ hay khó
   nhất?
2. Mô hình không dùng siêu dữ liệu sàng lọc bình thường/bất thường tốt đến đâu?
3. Sự đánh đổi giữa độ nhạy và độ đặc hiệu khi sàng lọc là gì?
4. Đặc trưng tín hiệu nào mang trọng số chẩn đoán lớn nhất, và có khớp với sinh lý
   tim đã biết không?

---

## 4. Phương pháp

### 4.1 Tiền xử lý
Mỗi bản ghi được lọc bằng **bộ lọc thông dải Butterworth bậc 4 (0,5-40 Hz, pha
không `filtfilt`)** để loại trôi đường nền và nhiễu tần số cao, rồi **chuẩn hóa
z-score theo từng chuyển đạo**.

![Hình 2: ECG đã tiền xử lý]({F}/ecg_sample.png)
*Hình 2: Một bản ghi đã tiền xử lý, cả 12 chuyển đạo, với các đỉnh R được phát hiện
(dấu x đỏ) trên chuyển đạo II.*

### 4.2 Trích xuất đặc trưng
Vector **128 đặc trưng** cố định mỗi bản ghi: **nhịp/HRV** từ đỉnh R chuyển đạo II;
**thống kê miền thời gian theo chuyển đạo** (độ lệch chuẩn, RMS, biên độ, độ lệch,
độ nhọn, tỷ lệ cắt không); và **công suất dải phổ theo chuyển đạo** (phương pháp
Welch trong bốn dải).

### 4.3 Chia dữ liệu
Phân vùng `strat_fold` chính thức của PTB-XL ngăn rò rỉ bệnh nhân:
**fold 1-8 huấn luyện ({ctx['n_tr']:,})**, **fold 9 kiểm định ({ctx['n_va']:,})**,
**fold 10 kiểm tra ({ctx['n_te']:,})**. Mọi kết quả bên dưới trên fold 10.

### 4.4 Mô hình
Logistic Regression, Random Forest (300 cây) và Histogram Gradient Boosting trong
quy trình `StandardScaler`, mỗi mô hình dùng `class_weight="balanced"`. Mô hình tốt
nhất trên tập kiểm định được giữ lại.

### 4.5 Chỉ số đánh giá
**ROC-AUC** (xếp hạng độc lập ngưỡng) là chỉ số chính; F1, độ chính xác, average
precision, độ nhạy và độ đặc hiệu là chỉ số bổ trợ.

---

## 5. Kết quả

Mô hình tốt nhất cho cả hai nhiệm vụ là **{ctx['model_ml']}**.

| Nhiệm vụ | Chỉ số chính |
|---------|-------------|
| 5 nhóm (đa nhãn) | macro-AUC **{ctx['macro_auc']:.3f}**, micro-AUC {ctx['micro_auc']:.3f} |
| Nhị phân (bình thường/bất thường) | ROC-AUC **{ctx['bi_auc']:.3f}**, độ chính xác {ctx['bi_acc']:.3f}, F1 {ctx['bi_f1']:.3f} |

### 5.1 Phân biệt đa nhãn

![Hình 3: ROC theo nhóm]({F}/multilabel_roc.png)
*Hình 3: Đường ROC theo từng nhóm chính (macro-AUC {ctx['macro_auc']:.3f}).*

Cả năm đường ROC nằm rõ trên mức ngẫu nhiên, khẳng định tín hiệu chẩn đoán thực sự
trong các đặc trưng thủ công. **{ctx['best_sc']}** dễ nhất
(AUC {ctx['per_auc'][ctx['best_sc']]:.3f}) và **{ctx['worst_sc']}** khó nhất
(AUC {ctx['per_auc'][ctx['worst_sc']]:.3f}).

| Nhóm chẩn đoán | ROC-AUC | F1 | Độ chính xác |
|--------------|---------|----|-------------|
{mlr}

![Hình 4: AUC so với F1 theo lớp]({F}/multilabel_per_class.png)
*Hình 4: ROC-AUC theo nhóm bên cạnh F1 tại ngưỡng 0,5.*

AUC vẫn cao trong khi **F1 giảm ở các lớp hiếm**, chính là khoảng cách do mất cân
bằng. Độ chính xác theo nhãn (Hamming) trung bình {ctx['hamming_acc']:.1%}, nhưng
đoán đúng cả năm nhãn chỉ xảy ra {ctx['subset_acc']:.1%} thời gian, nên AUC chứ
không phải độ chính xác mới là chỉ số chính công bằng.

### 5.2 Sàng lọc nhị phân

![Hình 5: Ma trận nhầm lẫn]({F}/binary_confusion.png)
*Hình 5: Ma trận nhầm lẫn bình thường/bất thường tại ngưỡng 0,5
(độ nhạy {ctx['sens']:.1%}, độ đặc hiệu {ctx['spec']:.1%}).*

Lỗi tốn kém là ô dưới-trái: **{ctx['fn']} âm tính giả**, ECG bất thường bị gọi là
bình thường. Điểm hiện tại cho độ nhạy {ctx['sens']:.1%} so với độ đặc hiệu
{ctx['spec']:.1%}.

![Hình 6: ROC và precision-recall]({F}/binary_roc_pr.png)
*Hình 6: ROC (AUC {ctx['bi_auc']:.3f}) và precision-recall
(average precision {ctx['bi_ap']:.3f}).*

Đường precision-recall là góc nhìn trung thực hơn khi mất cân bằng và nằm rõ trên
mức nền dương {ctx['binary_abnormal']:.2f}.

### 5.3 Tầm quan trọng đặc trưng

![Hình 7: Tầm quan trọng đặc trưng]({F}/feature_importance.png)
*Hình 7: 20 đặc trưng quan trọng nhất.*

Các đặc trưng giàu thông tin nhất gồm:

{_feat_lines(ctx)}

Chúng khớp với sinh lý dễ nhận biết: thống kê hình dạng theo chuyển đạo (độ lệch,
tỷ lệ cắt không) phản ánh hình thái QRS/T, công suất dải phổ nắm bắt nội dung tần
số, và các số hạng HRV nắm bắt nhịp.

---

## 6. Thảo luận

- **Tín hiệu chẩn đoán có thật nhưng không đồng đều.** macro-AUC
  {ctx['macro_auc']:.3f} chỉ từ đặc trưng là một mốc cổ điển mạnh; hiệu năng theo
  mức độ cục bộ của từng mẫu ({ctx['best_sc']} dễ nhất, {ctx['worst_sc']} khó nhất).
- **Mất cân bằng chi phối mọi thứ.** NORM chiếm đa số nên độ chính xác gây hiểu
  lầm và các lớp hiếm có khoảng cách AUC-F1, thúc đẩy điều chỉnh ngưỡng theo lớp.
- **Sàng lọc không cần siêu dữ liệu là khả thi.** AUC nhị phân {ctx['bi_auc']:.3f}
  với độ nhạy {ctx['sens']:.1%} là tín hiệu phân loại dùng được, không phải chẩn đoán.
- **Đặc trưng hợp lý về sinh lý**, giúp mô hình có thể kiểm tra được.

---

## 7. Hạn chế

1. Đặc trưng cổ điển bị giới hạn dưới học sâu (CNN đã công bố ~0,93 macro-AUC so với
   {ctx['macro_auc']:.3f} ở đây).
2. Chỉ phát hiện đỉnh R trên chuyển đạo II.
3. F1/độ chính xác báo cáo dùng ngưỡng cố định 0,5.
4. Chỉ dùng tín hiệu 100 Hz.
5. Khả năng của mã SCP bị thu gọn thành sự hiện diện nhị phân của nhóm.
6. Chỉ dùng cho học thuật; chưa được kiểm chứng lâm sàng.

---

## 8. Kết luận

Xử lý tín hiệu cổ điển thu được cấu trúc có ý nghĩa từ ECG 12 chuyển đạo:
**macro-AUC {ctx['macro_auc']:.3f}** cho năm nhóm chính và **ROC-AUC
{ctx['bi_auc']:.3f}** cho sàng lọc nhị phân trên fold kiểm tra tách biệt bệnh nhân,
với các đặc trưng diễn giải được về sinh lý. Nó không thể thay thế chuyên gia đọc
hay mô hình sâu, nhưng là một mốc minh bạch cho công cụ phân loại. Hướng tương lai:
điều chỉnh ngưỡng theo lớp, đặc trưng wavelet/mẫu nhịp, tín hiệu 500 Hz, và một mốc
CNN 1 chiều.

---

## Tài liệu tham khảo

Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation, 101*(23), e215-e220.

Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
*IEEE Transactions on Biomedical Engineering, BME-32*(3), 230-236.

Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning
for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE JBHI, 25*(5),
1519-1528.

Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography
dataset. *Scientific Data, 7*, 154.
"""


RENDERERS = {"REPORT.md": render_en, "report_th.md": render_th, "report_vn.md": render_vi}
