# 🌳 CÂY PHẢ HỆ DỮ LIỆU KAGGLE (KAGGLE DATASETS FAMILY TREE)
### Đề Tài: Phân Loại Phân Tử PAM50 & Tiên Lượng Sống Còn Ung Thư Vú Đa Phương Thức (TCGA-BRCA)

---

## 🗺️ 1. SƠ ĐỒ CÂY PHẢ HỆ TỔNG QUAN (GLOBAL DATASET ECOSYSTEM)

Toàn bộ hệ sinh thái dữ liệu của đề tài được xây dựng và liên kết chặt chẽ qua **5 Bộ Dataset trên Kaggle** (chia làm 2 nhóm: **3 Dataset Cá Nhân Tự Xuất Bản** và **2 Dataset Ảnh Khổng Lồ Cộng Đồng**):

```
                        ╔═══════════════════════════════════════════════╗
                        ║   NGUỒN DỮ LIỆU GỐC TCGA-BRCA (GDC NIH)       ║
                        ╚═══════════════════════════════════════════════╝
                                               │
                ┌──────────────────────────────┼──────────────────────────────┐
                ▼                              ▼                              ▼
    [ KHO TIÊU BẢN GỐC SVS ]       [ KHO PHIÊN MÃ HỌC RNA-SEQ ]    [ KHO DỮ LIỆU LÂM SÀNG ]
 ammarshafiq/cancer-multi-omics-         trihuynhviprovcl/             trihuynhviprovcl/
       and-imaging-data                     tcga-brca                     tcga-brca
      (21.0 GB - .svs raw)             (20,531 gen - RSEM)          (1,082 bệnh nhân - 72 cột)
                │                              │                              │
                ▼ (Notebook 00 & 02)           ▼ (Notebook 06.0)              ▼ (Notebook 01)
    [ KHO MẢNH MÔ HỌC PATCHES ]     [ MA TRẬN 500 GEN CHUẨN HÓA ]   [ COHORT KHỚP NỐI 882 CA ]
 jmalagontorres/tcga-brca-survival-     trihuynhviprovcl/             trihuynhviprovcl/
            analysis                        tcga-brca                     tcga-brca
       (134.85 GB - .png)           (Z-score StandardScaler)     (Master Matched Clinical CSV)
                │                              │                              │
                ▼ (Backbone ResNet50)          │                              │
 ╔══════════════════════════════════╗          │                              │
 ║ trihuynhviprovcl/                ║          │                              │
 ║ tcga-brca-resnet50-wsi-features  ║          │                              │
 ║ (3.0 GB - pt_files/ 2048D)       ║          │                              │
 ╚══════════════════════════════════╝          │                              │
                │                              │                              │
                └──────────────┬───────────────┴──────────────────────────────┘
                               ▼ (Huấn luyện Mô hình Đa phương thức: Notebook 06.2, 06.6)
 ╔════════════════════════════════════════════════════════════════════════════╗
 ║ trihuynhviprovcl/tcga-brca-multimodal-genomics-weights                     ║
 ║ (68.0 MB - multimodal_genomics_fold1_best.pth ... fold5_best.pth)          ║
 ╚════════════════════════════════════════════════════════════════════════════╝
                               │
                               ▼ (Đóng gói & Phục vụ Web Demo: Notebook 10 & 11)
 ╔════════════════════════════════════════════════════════════════════════════╗
 ║ ỨNG DỤNG WEB DEMO CDSS (Clinical Decision Support System - Streamlit)     ║
 ╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 📂 2. CHI TIẾT TỪNG DATASET & ĐƯỜNG DẪN KAGGLE DIRECTORY

### 🧬 PHẦN A: 3 DATASET CỐT LÕI (DO TÁC GIẢ TRI HUYNH XUẤT BẢN)

#### 1. Dataset Lâm Sàng & Phiên Mã Gen: `tcga-brca` (Version 4)
- **Link Kaggle:** [https://www.kaggle.com/datasets/trihuynhviprovcl/tcga-brca](https://www.kaggle.com/datasets/trihuynhviprovcl/tcga-brca)
- **Tác giả:** Tri Huynh (`trihuynhviprovcl`)
- **Dung lượng:** **166.4 MB** (Version 4)
- 📍 **Đường dẫn trên Kaggle (Kaggle Directory):**
  ```text
  /kaggle/input/datasets/trihuynhviprovcl/tcga-brca/
  ```
- **Cấu trúc cây thư mục & Đường dẫn file chi tiết:**
  ```text
  /kaggle/input/datasets/trihuynhviprovcl/tcga-brca/
  ├── data_mrna_seq_v2_rsem.txt             # Ma trận biểu hiện 20,531 gen RSEM normalized của 1,082 bệnh nhân
  ├── gdc_brca_clinical_treatments.csv      # Dữ liệu phác đồ điều trị (Hóa trị, Xạ trị, Thuốc nội tiết, Thuốc đích)
  ├── tcga_brca_master_matched_cohort.csv   # File Master lâm sàng đã khớp nối 882 bệnh nhân đủ 3 kênh (72 cột)
  └── tcga_brca_rna_500_scaled.csv          # Ma trận Top 500 gen đã chuẩn hóa Z-score (StandardScaler)
  ```
- **Đoạn mã Python khai báo đường dẫn chuẩn trong Notebook:**
  ```python
  CLINICAL_CSV = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv'
  RNASEQ_RAW   = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/data_mrna_seq_v2_rsem.txt'
  RNASEQ_500   = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_rna_500_scaled.csv'
  TREATMENTS   = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/gdc_brca_clinical_treatments.csv'
  ```
- **Vai trò y sinh:** Cung cấp toàn bộ nhãn giải phẫu bệnh PAM50 ground truth, các thông số sống còn (`survival_months`, `censored`) và nồng độ phiên mã của 20,531 gen (đặc biệt là các gen ung thư then chốt: `ERBB2`, `ESR1`, `PGR`, `MKI67`, `TP53`, `BRCA1/2`).
- **Sử dụng trong:** Notebook `01`, `03.2`, `03.3`, `06.0`, `06.1`, `06.2`, `06.4`, `06.6`, `07`, `07.1`, `08`, `09`, `10`.

---

#### 2. Dataset Trọng Số Mô Hình: `tcga-brca-multimodal-genomics-weights`
- **Link Kaggle:** [https://www.kaggle.com/datasets/trihuynhviprovcl/tcga-brca-multimodal-genomics-weights](https://www.kaggle.com/datasets/trihuynhviprovcl/tcga-brca-multimodal-genomics-weights)
- **Tác giả:** Tri Huynh (`trihuynhviprovcl`)
- **Dung lượng:** **68.0 MB**
- 📍 **Đường dẫn trên Kaggle (Kaggle Directory):**
  ```text
  /kaggle/input/datasets/trihuynhviprovcl/tcga-brca-multimodal-genomics-weights/
  ```
- **Cấu trúc cây thư mục & Đường dẫn file chi tiết:**
  ```text
  /kaggle/input/datasets/trihuynhviprovcl/tcga-brca-multimodal-genomics-weights/
  ├── multimodal_genomics_fold1_best.pth     # Checkpoint PyTorch tốt nhất Fold 1 (TransMIL God Mode)
  ├── multimodal_genomics_fold2_best.pth     # Checkpoint PyTorch Fold 2
  ├── multimodal_genomics_fold3_best.pth     # Checkpoint PyTorch Fold 3 (F1 cao nhất)
  ├── multimodal_genomics_fold4_best.pth     # Checkpoint PyTorch Fold 4
  └── multimodal_genomics_fold5_best.pth     # Checkpoint PyTorch Fold 5
  ```
- **Đoạn mã Python khai báo đường dẫn chuẩn trong Notebook:**
  ```python
  WEIGHTS_DIR  = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-multimodal-genomics-weights'
  FOLD1_WEIGHT = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-multimodal-genomics-weights/multimodal_genomics_fold1_best.pth'
  ```
- **Vai trò y sinh:** Lưu giữ "bộ não" đã được tối ưu hóa qua 5-Fold Cross-Validation của Siêu mô hình Đa phương thức (TransMIL 512D + Genomics MLP 512D $\to$ 1024D). Phục vụ suy luận siêu tốc (Inference), giải thích XAI và chạy Web App offline mà không cần train lại.
- **Sử dụng trong:** Notebook `06.4`, `07`, `08`, `09`, `10`, `Web Demo CDSS`.

---

#### 3. Dataset Tensor Đặc Trưng Hình Ảnh WSI: `tcga-brca-resnet50-wsi-features`
- **Link Kaggle:** [https://www.kaggle.com/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features](https://www.kaggle.com/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features)
- **Tác giả:** Tri Huynh (`trihuynhviprovcl`)
- **Dung lượng:** **3.0 GB**
- 📍 **Đường dẫn trên Kaggle (Kaggle Directory):**
  ```text
  /kaggle/input/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features/
  ```
- **Cấu trúc cây thư mục & Đường dẫn file chi tiết:**
  ```text
  /kaggle/input/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features/
  └── pt_files/                              # Thư mục chứa 882 file tensor PyTorch (.pt)
      ├── TCGA-3C-AAAU.pt                    # Tensor kích thước [N_patches x 2048]
      ├── TCGA-3C-AALI.pt
      ├── ...
      ├── TCGA-OL-A66J.pt                    # Tensor mẫu phân nhóm Luminal A
      ├── TCGA-A8-A08P.pt                    # Tensor mẫu phân nhóm Luminal B
      ├── TCGA-AQ-A04J.pt                    # Tensor mẫu phân nhóm Basal-like
      ├── TCGA-C8-A12Z.pt                    # Tensor mẫu phân nhóm HER2-enriched
      └── TCGA-Z7-A8R6.pt
  ```
- **Đoạn mã Python khai báo đường dẫn chuẩn trong Notebook:**
  ```python
  PT_DIR = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features/pt_files'
  ```
- **Vai trò y sinh:** Mỗi file `.pt` đại diện cho một tiêu bản mô bệnh học WSI hoàn chỉnh của 1 bệnh nhân, chứa $N$ vector 2048 chiều (tương ứng với hàng trăm/hàng ngàn mảnh tế bào vi thể $256\times 256$ trích xuất qua mạng CNN ResNet50).
- **Sử dụng trong:** Notebook `02`, `03`, `03.1`, `03.2`, `03.3`, `04`, `05`, `05.1`, `06.1`, `06.2`, `06.4`, `06.6`, `07`, `07.1`, `08`, `10`.

---

### 🖼️ PHẦN B: 2 DATASET ẢNH KHỔNG LỒ (TÀI NGUYÊN CỘNG ĐỒNG)

#### 4. Dataset Mảnh Ảnh Vi Thể (Patches): `tcga-brca-survival-analysis`
- **Link Kaggle:** [https://www.kaggle.com/datasets/jmalagontorres/tcga-brca-survival-analysis](https://www.kaggle.com/datasets/jmalagontorres/tcga-brca-survival-analysis)
- **Tác giả:** J. Malagon-Torres
- **Dung lượng:** **134.85 GB**
- 📍 **Đường dẫn trên Kaggle (Kaggle Directory):**
  ```text
  /kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/
  ```
- **Cấu trúc cây thư mục & Đường dẫn file chi tiết:**
  ```text
  /kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/
  ├── clinical_data(labels).csv              # Bảng dữ liệu sống còn và nhãn phân loại
  └── WSIs/                                  # Thư mục chứa các folder ảnh patch của từng bệnh nhân
      ├── TCGA-OL-A66J/                      # Thư mục ảnh vi thể bệnh nhân TCGA-OL-A66J
      │   ├── TCGA-OL-A66J_001.png           # Ảnh mô học nhuộm H&E (256 x 256 pixels)
      │   ├── TCGA-OL-A66J_002.png
      │   └── ...
      ├── TCGA-A8-A08P/
      ├── TCGA-AQ-A04J/
      └── TCGA-C8-A12Z/
  ```
- **Đoạn mã Python khai báo đường dẫn chuẩn trong Notebook:**
  ```python
  PATCHES_ROOT = '/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/WSIs'
  ```
- **Vai trò y sinh:** Cung cấp các bức ảnh tế bào học H&E thực tế (mắt thường và kính hiển vi nhìn thấy). Dùng để huấn luyện mô hình trích xuất đặc trưng CNN (Notebook 02), trích xuất ảnh mẫu phục vụ vẽ thảm ảnh **Mosaic Attention viền màu** trên Web Demo (Notebook 11).
- **Sử dụng trong:** Notebook `00`, `02`, `11`, `Web Demo Mosaic Viewer`.

---

#### 5. Dataset Tiêu Bản Kính Hiển Vi Nguyên Bản (Gigapixel SVS): `cancer-multi-omics-and-imaging-data`
- **Link Kaggle:** [https://www.kaggle.com/datasets/ammarshafiq/cancer-multi-omics-and-imaging-data](https://www.kaggle.com/datasets/ammarshafiq/cancer-multi-omics-and-imaging-data)
- **Tác giả:** Ammar Shafiq
- **Dung lượng:** **21.0 GB**
- 📍 **Đường dẫn trên Kaggle (Kaggle Directory):**
  ```text
  /kaggle/input/datasets/ammarshafiq/cancer-multi-omics-and-imaging-data/
  ```
- **Cấu trúc cây thư mục & Đường dẫn file chi tiết:**
  ```text
  /kaggle/input/datasets/ammarshafiq/cancer-multi-omics-and-imaging-data/
  └── ... /                                  # Chứa các file tiêu bản kính hiển vi Whole Slide Images
      ├── TCGA-4P-AA8J.svs                   # File SVS cấu trúc kim tự tháp (Pyramid 4 tầng)
      ├── TCGA-CR-6478.svs                   # File SVS nhẹ (~6.05 MB)
      ├── TCGA-CR-7391.svs                   # File SVS nhẹ (~6.79 MB)
      └── ...
  ```
- **Đoạn mã Python khai báo đường dẫn chuẩn trong Notebook:**
  ```python
  SVS_ROOT = '/kaggle/input/datasets/ammarshafiq/cancer-multi-omics-and-imaging-data'
  ```
- **Vai trò y sinh:** Chứa các file ảnh siêu trường nhìn (Gigapixel WSI, độ phân giải lên tới $100,000 \times 30,000$ pixels ở độ phóng đại $40\times$). Dùng cho bài toán tiền xử lý phân đoạn mô Otsu & Color Deconvolution (Notebook 00) và trình diễn tính năng Kính hiển vi tương tác (Interactive Zoom Viewer) trên Web Demo (Notebook 11).
- **Sử dụng trong:** Notebook `00`, `11`, `Web Demo SVS Viewer`.

---

## 📊 3. MA TRẬN PHỐI HỢP DATASET GIỮA CÁC NOTEBOOK (CROSS-REFERENCE MATRIX)

Bảng dưới đây chỉ rõ khi chạy bất kỳ Notebook nào trên Kaggle, bạn cần **Add đúng những Dataset nào** vào môi trường:

| Mã Notebook | Tên Notebook | `tcga-brca` (166MB) | `genomics-weights` (68MB) | `resnet50-features` (3GB) | `survival-patches` (135GB) | `multi-omics-svs` (21GB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **00** | `00-wsi-svs-to-patches-preprocessing` | ❌ | ❌ | ❌ | ❌ | ✅ **Cần** |
| **01** | `01-tcga-eda-and-clinical-processing` | ✅ **Cần** | ❌ | ❌ | ❌ | ❌ |
| **02** | `02-cnn-feature-extraction-and-baseline` | ✅ **Cần** | ❌ | ❌ | ✅ **Cần** | ❌ |
| **03** | `03-classification-traditional-ml` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **03.2** | `03-2-classification-ml-clinical-fusion` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **03.3** | `03-3-classification-ml-genomics-fusion` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **04** | `04-classification-naive-deepmil` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **05** | `05-classification-transmil-sota` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **05.1** | `05-1-classification-transmil-ablation` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **06.0** | `06-0-multimodal-eda-and-genomics` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **06.1** | `06-1-multimodal-wsi-clinical` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **06.2** | `06-2-multimodal-wsi-genomics` (SOTA) | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **06.4** | `06-4-multimodal-architecture-deepdive`| ✅ **Cần** | ✅ **Cần** | ✅ **Cần** | ❌ | ❌ |
| **06.6** | `06-6-multimodal-wsi-bilstm-gen` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **07** | `07-survival-analysis-multimodal` | ✅ **Cần** | ✅ **Cần** | ✅ **Cần** | ❌ | ❌ |
| **07.1** | `07-1-lstm-survival-prediction` | ✅ **Cần** | ❌ | ✅ **Cần** | ❌ | ❌ |
| **08** | `08-explainable-ai-multimodal` (XAI) | ✅ **Cần** | ✅ **Cần** | ✅ **Cần** | ❌ | ❌ |
| **09** | `09-digital-twin-treatment-simulation` | ✅ **Cần** | ✅ **Cần** | ✅ **Cần** | ❌ | ❌ |
| **10** | `10-prepare-web-demo-assets-package` | ✅ **Cần** | ✅ **Cần** | ✅ **Cần** | ❌ | ❌ |
| **11** | `11-extract-wsi-patches-and-svs` | ❌ | ❌ | ❌ | ✅ **Cần** | ✅ **Cần** |

---

## 🎯 4. TỔNG KẾT BẢO LƯU & SAO LƯU (BEST PRACTICES)

1. **Bộ 3 Dataset Cá Nhân (`tcga-brca`, `weights`, `resnet50-features`):** Tổng dung lượng chỉ ~3.23 GB, là tài sản cốt lõi của đề tài, chứa toàn bộ dữ liệu đã được làm sạch, trích xuất đặc trưng và trọng số tối ưu.
2. **Bộ 2 Dataset Ảnh Cộng Đồng (`patches 135GB`, `svs 21GB`):** Chỉ cần gắn vào khi chạy Notebook 00, 02 (khâu tiền xử lý) hoặc Notebook 11 (khâu bốc tách ảnh mẫu cho Web Demo). Khi chạy các mô hình AI chính, chỉ cần dùng vector `.pt` (3GB) để tiết kiệm thời gian nạp và tối ưu băng thông.
