# Sàng lọc rối loạn nhịp tim tự động từ ECG 12 chuyển đạo với bộ dữ liệu PTB-XL

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

Sau khi loại các bản ghi không có nhóm chẩn đoán, còn lại **21,388** bản
ghi. Dữ liệu mất cân bằng; tổng số đếm vượt tổng bản ghi vì nhãn là đa nhãn.

| Nhóm chẩn đoán | Số lượng | Tỷ lệ |
|--------------|---------|-------|
| NORM (ECG bình thường) | 9,514 | 44.5% |
| MI (Nhồi máu cơ tim) | 5,469 | 25.6% |
| STTC (Thay đổi ST/T) | 5,235 | 24.5% |
| CD (Rối loạn dẫn truyền) | 4,898 | 22.9% |
| HYP (Phì đại) | 2,649 | 12.4% |

Một bản ghi là **bình thường** chỉ khi nhóm duy nhất của nó là NORM, cho
9,069 bản ghi bình thường và 12,319 bất thường
(57.6% bất thường).

![Hình 1: Tỷ lệ các lớp](reports/figures/eda_class_prevalence.png)
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

![Hình 2: ECG đã tiền xử lý](reports/figures/ecg_sample.png)
*Hình 2: Một bản ghi đã tiền xử lý, cả 12 chuyển đạo, với các đỉnh R được phát hiện
(dấu x đỏ) trên chuyển đạo II.*

### 4.2 Trích xuất đặc trưng
Vector **128 đặc trưng** cố định mỗi bản ghi: **nhịp/HRV** từ đỉnh R chuyển đạo II;
**thống kê miền thời gian theo chuyển đạo** (độ lệch chuẩn, RMS, biên độ, độ lệch,
độ nhọn, tỷ lệ cắt không); và **công suất dải phổ theo chuyển đạo** (phương pháp
Welch trong bốn dải).

### 4.3 Chia dữ liệu
Phân vùng `strat_fold` chính thức của PTB-XL ngăn rò rỉ bệnh nhân:
**fold 1-8 huấn luyện (17,084)**, **fold 9 kiểm định (2,146)**,
**fold 10 kiểm tra (2,158)**. Mọi kết quả bên dưới trên fold 10.

### 4.4 Mô hình
Logistic Regression, Random Forest (300 cây) và Histogram Gradient Boosting trong
quy trình `StandardScaler`, mỗi mô hình dùng `class_weight="balanced"`. Mô hình tốt
nhất trên tập kiểm định được giữ lại.

### 4.5 Chỉ số đánh giá
**ROC-AUC** (xếp hạng độc lập ngưỡng) là chỉ số chính; F1, độ chính xác, average
precision, độ nhạy và độ đặc hiệu là chỉ số bổ trợ.

---

## 5. Kết quả

Mô hình tốt nhất cho cả hai nhiệm vụ là **hist_gbdt**.

| Nhiệm vụ | Chỉ số chính |
|---------|-------------|
| 5 nhóm (đa nhãn) | macro-AUC **0.859**, micro-AUC 0.884 |
| Nhị phân (bình thường/bất thường) | ROC-AUC **0.894**, độ chính xác 0.811, F1 0.830 |

### 5.1 Phân biệt đa nhãn

![Hình 3: ROC theo nhóm](reports/figures/multilabel_roc.png)
*Hình 3: Đường ROC theo từng nhóm chính (macro-AUC 0.859).*

Cả năm đường ROC nằm rõ trên mức ngẫu nhiên, khẳng định tín hiệu chẩn đoán thực sự
trong các đặc trưng thủ công. **NORM** dễ nhất
(AUC 0.901) và **HYP** khó nhất
(AUC 0.797).

| Nhóm chẩn đoán | ROC-AUC | F1 | Độ chính xác |
|--------------|---------|----|-------------|
| NORM | 0.901 | 0.805 | 0.819 |
| STTC | 0.888 | 0.627 | 0.842 |
| CD | 0.871 | 0.657 | 0.866 |
| MI | 0.837 | 0.540 | 0.804 |
| HYP | 0.797 | 0.248 | 0.885 |

![Hình 4: AUC so với F1 theo lớp](reports/figures/multilabel_per_class.png)
*Hình 4: ROC-AUC theo nhóm bên cạnh F1 tại ngưỡng 0,5.*

AUC vẫn cao trong khi **F1 giảm ở các lớp hiếm**, chính là khoảng cách do mất cân
bằng. Độ chính xác theo nhãn (Hamming) trung bình 84.3%, nhưng
đoán đúng cả năm nhãn chỉ xảy ra 50.4% thời gian, nên AUC chứ
không phải độ chính xác mới là chỉ số chính công bằng.

### 5.2 Sàng lọc nhị phân

![Hình 5: Ma trận nhầm lẫn](reports/figures/binary_confusion.png)
*Hình 5: Ma trận nhầm lẫn bình thường/bất thường tại ngưỡng 0,5
(độ nhạy 80.0%, độ đặc hiệu 82.6%).*

Lỗi tốn kém là ô dưới-trái: **249 âm tính giả**, ECG bất thường bị gọi là
bình thường. Điểm hiện tại cho độ nhạy 80.0% so với độ đặc hiệu
82.6%.

![Hình 6: ROC và precision-recall](reports/figures/binary_roc_pr.png)
*Hình 6: ROC (AUC 0.894) và precision-recall
(average precision 0.929).*

Đường precision-recall là góc nhìn trung thực hơn khi mất cân bằng và nằm rõ trên
mức nền dương 0.58.

### 5.3 Tầm quan trọng đặc trưng

![Hình 7: Tầm quan trọng đặc trưng](reports/figures/feature_importance.png)
*Hình 7: 20 đặc trưng quan trọng nhất.*

Các đặc trưng giàu thông tin nhất gồm:

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

Chúng khớp với sinh lý dễ nhận biết: thống kê hình dạng theo chuyển đạo (độ lệch,
tỷ lệ cắt không) phản ánh hình thái QRS/T, công suất dải phổ nắm bắt nội dung tần
số, và các số hạng HRV nắm bắt nhịp.

---

## 6. Thảo luận

- **Tín hiệu chẩn đoán có thật nhưng không đồng đều.** macro-AUC
  0.859 chỉ từ đặc trưng là một mốc cổ điển mạnh; hiệu năng theo
  mức độ cục bộ của từng mẫu (NORM dễ nhất, HYP khó nhất).
- **Mất cân bằng chi phối mọi thứ.** NORM chiếm đa số nên độ chính xác gây hiểu
  lầm và các lớp hiếm có khoảng cách AUC-F1, thúc đẩy điều chỉnh ngưỡng theo lớp.
- **Sàng lọc không cần siêu dữ liệu là khả thi.** AUC nhị phân 0.894
  với độ nhạy 80.0% là tín hiệu phân loại dùng được, không phải chẩn đoán.
- **Đặc trưng hợp lý về sinh lý**, giúp mô hình có thể kiểm tra được.

---

## 7. Hạn chế

1. Đặc trưng cổ điển bị giới hạn dưới học sâu (CNN đã công bố ~0,93 macro-AUC so với
   0.859 ở đây).
2. Chỉ phát hiện đỉnh R trên chuyển đạo II.
3. F1/độ chính xác báo cáo dùng ngưỡng cố định 0,5.
4. Chỉ dùng tín hiệu 100 Hz.
5. Khả năng của mã SCP bị thu gọn thành sự hiện diện nhị phân của nhóm.
6. Chỉ dùng cho học thuật; chưa được kiểm chứng lâm sàng.

---

## 8. Kết luận

Xử lý tín hiệu cổ điển thu được cấu trúc có ý nghĩa từ ECG 12 chuyển đạo:
**macro-AUC 0.859** cho năm nhóm chính và **ROC-AUC
0.894** cho sàng lọc nhị phân trên fold kiểm tra tách biệt bệnh nhân,
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
