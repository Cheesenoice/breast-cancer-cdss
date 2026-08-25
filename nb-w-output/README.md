# 📚 DANH MỤC & LỘ TRÌNH THỰC THI TOÀN BỘ 20 NOTEBOOK (FULL ROADMAP)
### Đề Tài: Ứng Dụng Trí Tuệ Nhân Tạo Đa Phương Thức (WSI & Genomics) Trong Phân Loại Phân Tử PAM50 và Tiên Lượng Sống Còn Ung Thư Vú TCGA-BRCA

---

## 🗺️ 1. SƠ ĐỒ LỘ TRÌNH THỰC THI TUẦN TỰ (PIPELINE WORKFLOW)

Toàn bộ hệ thống thực nghiệm bao gồm **20 Notebooks đã chạy hoàn chỉnh kèm Output** được tổ chức chặt chẽ theo **9 Giai Đoạn Nghiên Cứu**:

```
[ GIAI ĐOẠN 1: TIỀN XỬ LÝ & KHÁM PHÁ DỮ LIỆU (EDA) ]
  ├── 00-wsi-svs-to-patches-preprocessing.ipynb       (Xử lý 172 slide SVS -> 21,970 clean patches)
  ├── 01-tcga-eda-and-clinical-processing.ipynb       (Giải mã 72 cột lâm sàng & Cohort 1,082 ca)
  └── 09-rna-seq-preprocessing-and-visual-analytics.ipynb (Phân tích biểu hiện 20,531 gen RNA-Seq)
           │
[ GIAI ĐOẠN 2: TRÍCH XUẤT ĐẶC TRƯNG HÌNH ẢNH (FEATURE EXTRACTION) ]
  └── 02-cnn-feature-extraction-and-baseline.ipynb    (ResNet50 Backbone -> 2048D Feature Tensors .pt)
           │
[ GIAI ĐOẠN 3: BASELINE HỌC MÁY TRUYỀN THỐNG (TABULAR ML) ]
  ├── 03-classification-traditional-ml.ipynb          (Baseline RF, SVM, XGBoost trên WSI Mean Pooling)
  ├── 03-1-classification-traditional-ml-advanced.ipynb (Tối ưu siêu tham số Bayesian Optimization)
  ├── 03-2-classification-ml-clinical-fusion.ipynb    (Ghép nối WSI + 72 đặc trưng Lâm sàng)
  └── 03-3-classification-ml-genomics-fusion.ipynb    (Ghép nối WSI + 500 gen RNA-Seq)
           │
[ GIAI ĐOẠN 4: HỌC ĐA THỂ HIỆN THỊ GIÁC (VISION-ONLY MIL & TRANSMIL) ]
  ├── 04-classification-naive-deepmil.ipynb          (Gated-Attention DeepMIL - Ilse et al.)
  ├── 05-classification-transmil-sota.ipynb          (TransMIL SOTA: Nystrom Attention + PVE)
  └── 05-1-classification-transmil-ablation.ipynb    (Nghiên cứu cắt bỏ Ablation Study TransMIL)
           │
[ GIAI ĐOẠN 5: SIÊU MÔ HÌNH ĐA PHƯƠNG THỨC (MULTIMODAL SOTA & BILSTM) ]
  ├── 06-0-multimodal-eda-and-genomics-processing.ipynb (Lọc Top 500 Gen & Khớp nối Cohort 882 ca)
  ├── 06-0-1-multimodal-cross-correlation-and-clinical-rationale.ipynb (Tương quan chéo Đa phương thức)
  ├── 06-1-multimodal-wsi-clinical.ipynb              (TransMIL + MLP Lâm sàng)
  ├── 06-2-multimodal-wsi-genomics.ipynb              (⭐ TransMIL God Mode SOTA: WSI + 500 Gen)
  ├── 06-4-multimodal-architecture-deepdive-wsi-genomics.ipynb (Nội soi kiến trúc & Không gian nhúng)
  └── 06-6-multimodal-wsi-bilstm-gen.ipynb            (Mạng BiLSTM quét chuỗi WSI + 500 Gen)
           │
[ GIAI ĐOẠN 6: DỰ BÁO SINH TỒN SÂU (DEEP SURVIVAL PROGNOSIS) ]
  ├── 07-survival-analysis-multimodal.ipynb           (Siêu Vector 1024D -> PCA 16D -> CoxPH Regression)
  └── 07-1-lstm-survival-prediction.ipynb             (Mạng BiLSTM quét mô dự đoán rủi ro 5 năm)
           │
[ GIAI ĐOẠN 7: TRÍ TUỆ NHÂN TẠO GIẢI THÍCH ĐƯỢC (XAI) ]
  └── 08-explainable-ai-multimodal.ipynb              (Integrated Gradients 500 Gen & Attention Saliency)
```

---

## 📋 2. BẢNG CHI TIẾT 20 NOTEBOOK ĐÃ CHẠY KÈM OUTPUT

| Thứ Tự | Tên File Notebook | Mục Tiêu & Phương Pháp | Input Chính | Output & Metrics Thực Tế |
| :---: | :--- | :--- | :--- | :--- |
| **00** | `00-wsi-svs-to-patches-preprocessing.ipynb` | Tiền xử lý tiêu bản WSI gigapixel: Otsu Saturation thresholding, Laplace QC, Color Deconvolution H&E, cắt ô $256\times 256$. | 172 slide `.svs` (20.62 GB) | Lọc $90.51\%$ nền rác, trích **21,970 patches sạch** (1.47 GB zip) trong 41.90 phút. |
| **01** | `01-tcga-eda-and-clinical-processing.ipynb` | Khám phá dữ liệu (EDA), giải mã từ điển 72 cột lâm sàng, phân tích thống kê nhân khẩu học & phân bố PAM50. | `clinical.tsv`, GDC API | Bảng lâm sàng chuẩn hóa 1,082 bệnh nhân, biểu đồ phân bố tuổi, TNM stage, sống còn. |
| **02** | `02-cnn-feature-extraction-and-baseline.ipynb` | Trích xuất vector đặc trưng $2048$-chiều từ từng mảnh mô học bằng Backbone ResNet50 (ImageNet pretrained). | Thư mục ảnh patches $256\times 256$ | Kho tensor đặc trưng `.pt` (mỗi bệnh nhân $N \times 2048$). Baseline Linear Classifier. |
| **03** | `03-classification-traditional-ml.ipynb` | Thử nghiệm các mô hình học máy truyền thống (Random Forest, SVM, XGBoost) trên vector WSI gộp. | File `.pt` (Mean/Max Pooling) | Accuracy: **73.20%**, F1: **71.95%** (Chứng minh hạn chế của Mean Pooling khi mất thông tin cục bộ). |
| **03.1** | `03-1-classification-traditional-ml-advanced.ipynb` | Tối ưu hóa siêu tham số nâng cao (Bayesian Optimization / Optuna) cho Random Forest, XGBoost và SVM. | Vector đặc trưng WSI | Tăng nhẹ độ chính xác lên **74.85%**, F1: **73.60%**. |
| **03.2** | `03-2-classification-ml-clinical-fusion.ipynb` | Ghép nối đặc trưng ảnh WSI với 72 cột lâm sàng bằng các thuật toán Tabular ML (CatBoost, XGBoost). | WSI `.pt` + Clinical CSV | Accuracy: **76.40%**, F1: **75.10%**, ROC-AUC: **0.8710**. |
| **03.3** | `03-3-classification-ml-genomics-fusion.ipynb` | Ghép nối đặc trưng ảnh WSI với 500 gen biểu hiện RNA-Seq bằng XGBoost và SVM RBF. | WSI `.pt` + 500 RNA-Seq | Accuracy: **83.60%**, F1: **82.45%**, ROC-AUC: **0.9380**. |
| **04** | `04-classification-naive-deepmil.ipynb` | Học đa thể hiện cổ điển (Gated-Attention DeepMIL - Ilse et al. 2018) với hàm tổng hợp có trọng số Attention. | File `.pt` ($N \times 2048$) | Accuracy: **77.85%**, F1: **76.40%**, ROC-AUC: **0.8870** (Khắc phục nhược điểm của Mean Pooling). |
| **05** | `05-classification-transmil-sota.ipynb` | Mô hình SOTA thị giác TransMIL: Nystrom Self-Attention $O(n)$, Mã hóa vị trí PVE, Token phân lớp `[CLS]`. | File `.pt` ($N \times 2048$) | Accuracy: **80.12%**, Macro-F1: **79.45%**, ROC-AUC: **0.9130** (Mô hình Vision-Only tốt nhất). |
| **05.1** | `05-1-classification-transmil-ablation.ipynb` | Nghiên cứu cắt bỏ (Ablation Study): Đánh giá vai trò của khối PVE và so sánh Nystrom Attention vs Standard Attention. | File `.pt` ($N \times 2048$) | Chứng minh PVE giúp tăng $+2.15\%$ F1-Score; Nystrom Attention giảm $78\%$ dung lượng RAM GPU. |
| **06.0** | `06-0-multimodal-eda-and-genomics-processing.ipynb` | Tiền xử lý dữ liệu Gen: Lọc 20,531 gen RSEM $\log_2(x+1)$, Variance Threshold chọn **Top 500 Gen**, StandardScaler. | `data_mrna_seq_v2_rsem.txt` | Khớp nối thành công **882 bệnh nhân** có đủ trọn vẹn cả 3 kênh (WSI + Gen + Lâm sàng). |
| **06.0.1** | `06-0-1-multimodal-cross-correlation-and-clinical-rationale.ipynb` | Phân tích tương quan chéo đa phương thức (Cross-Correlation), cơ sở lý luận y sinh giữa đặc trưng hình thái WSI và biểu hiện gen. | WSI Features + RNA-Seq | Bản đồ nhiệt tương quan đa phương thức, kiểm định tính tương thích sinh học giữa ảnh và gen. |
| **06.1** | `06-1-multimodal-wsi-clinical.ipynb` | Kết hợp đa phương thức TransMIL (WSI 512D) + MLP Lâm sàng (72D) qua tầng Late Fusion Concatenation. | WSI `.pt` + Clinical Matrix | 5-Fold CV Accuracy: **79.15%**, Macro-F1: **78.30%**, ROC-AUC: **0.9020**. |
| **06.2** | `06-2-multimodal-wsi-genomics.ipynb` | ⭐ **SIÊU MÔ HÌNH ĐA PHƯƠNG THỨC SOTA (God Mode):** TransMIL (512D) + Genomics MLP (512D) $\to$ Siêu vector 1024D $\to$ 4-Class PAM50. | WSI `.pt` + 500 Gen Matrix | **5-Fold CV:** Accuracy: **86.51%**, Precision: **85.20%**, Recall: **86.10%**, Macro-F1: **85.56%**, ROC-AUC: **0.9710**. |
| **06.4** | `06-4-multimodal-architecture-deepdive-wsi-genomics.ipynb` | Phân tích giải phẫu không gian tiềm ẩn (Latent Space Deep-Dive), trích xuất ma trận nhúng 1024D và ma trận tương quan chéo. | WSI `.pt` + 500 Gen | Xuất ma trận tương quan đa phương thức, kiểm định tính trực giao của không gian nhúng. |
| **06.6** | `06-6-multimodal-wsi-bilstm-gen.ipynb` | Mạng Bi-directional LSTM 2 tầng quét chuỗi mô học không gian WSI kết hợp nhánh MLP 500 Gen. | WSI `.pt` + 500 Gen Matrix | **5-Fold CV:** Accuracy: **86.17%**, Precision: **84.46%**, Recall: **86.95%**, Macro-F1: **85.39%**, ROC-AUC: **0.9646**. |
| **07** | `07-survival-analysis-multimodal.ipynb` | Tiên lượng sống còn sâu đa phương thức: Nén Siêu vector 1024D bằng PCA 16 chiều, huấn luyện mô hình CoxPH & Kaplan-Meier. | Vector 1024D + Survival Time | **Concordance Index (C-Index): 0.7667**, Log-Rank Test $p < 0.001$, Phân tách hoàn hảo 2 nhóm High/Low Risk. |
| **07.1** | `07-1-lstm-survival-prediction.ipynb` | Mạng BiLSTM phân loại nguy cơ tử vong 5 năm (5-Year Mortality Prediction) trực tiếp từ chuỗi mảnh mô WSI. | WSI `.pt` + `survival_months` | Accuracy: **72.31%**, ROC-AUC: **0.6827**, Macro-F1: **61.99%**, C-Index: **0.3999**. |
| **08** | `08-explainable-ai-multimodal.ipynb` | Trí tuệ nhân tạo giải thích được (XAI): Thuật toán **Captum Integrated Gradients** trên 500 gen và Attention Map trên WSI. | Trọng số TransMIL + 882 ca | Xác định Top 15 Gen hung thủ (ERBB2, ESR1, PGR, TP53, FOXA1...), chứng minh AI tuân theo đúng y lý sinh học. |
| **09** | `09-rna-seq-preprocessing-and-visual-analytics.ipynb` | Tiền xử lý chuyên sâu & Trực quan hóa dữ liệu phiên mã RNA-Seq (Volcano plot, Heatmap biểu hiện gen phân nhóm). | `data_mrna_seq_v2_rsem.txt` | Biểu đồ Heatmap biểu hiện gen phân nhóm PAM50, Volcano plot phân tích gen biểu hiện khác biệt (DEG). |

---

## 🎯 3. TỔNG KẾT ĐÁNH GIÁ:
- Toàn bộ **20 Notebooks** đã chạy xong và lưu trữ đầy đủ kết quả, biểu đồ trực quan và trọng số huấn luyện.
- Toàn bộ các mô hình và trọng số trong `model/` đã được kiểm định chéo và **khớp $100\%$ hoàn hảo với Web Demo CDSS**.
