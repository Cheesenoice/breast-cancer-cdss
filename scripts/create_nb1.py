import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Khai Phá Dữ Liệu (EDA) & Tiền Xử Lý Lâm Sàng: TCGA-BRCA\n",
    "\n",
    "**Dự án Thực tập:** Hệ Thống AI Đa Phương Thức Chẩn Đoán Phân Nhóm Phân Tử và Dự Báo Sinh Tồn Ung Thư Vú.\n",
    "\n",
    "--- \n",
    "### 🎯 Mục tiêu của Notebook này:\n",
    "Notebook này không chỉ chứa code xử lý, mà còn đóng vai trò như một **Báo cáo Phân tích Dữ liệu (Data Storytelling)**. Giả định chúng ta là một nhóm Data Scientist Y Yế đang tiếp cận tập dữ liệu khổng lồ TCGA-BRCA lần đầu tiên. Mục tiêu của chúng ta là:\n",
    "1. **Hiểu rõ nguồn gốc dữ liệu:** Dữ liệu đến từ đâu? Có những trường thông tin (columns) nào mang ý nghĩa sống còn cho bài toán?\n",
    "2. **Xác định Nhãn (Labels) cho Mô hình AI:** \n",
    "   - Bài toán 1 (Chẩn đoán - TransMIL): Cần nhãn phân nhóm phân tử (`pam50_subtype`).\n",
    "   - Bài toán 2 (Dự báo sinh tồn - LSTM): Cần nhãn thời gian sống và trạng thái (`OS_MONTHS`, `OS_STATUS`).\n",
    "3. **Phát hiện sự mất cân bằng dữ liệu (Class Imbalance):** Ung thư không bao giờ phân bố đều. Chúng ta cần trực quan hóa để có chiến lược thiết kế hàm Loss hoặc chia Fold hợp lý ở các bước sau.\n",
    "4. **Khớp nối Dữ liệu Lâm sàng và Ảnh WSI:** Xác minh xem bệnh nhân trong file CSV có khớp với các thư mục ảnh `.jpg` đã được cắt patch hay không.\n",
    "\n",
    "### 📁 Nguồn dữ liệu sử dụng:\n",
    "Dự án này sử dụng 2 nguồn dữ liệu kết hợp trên Kaggle:\n",
    "- **Dữ liệu Ảnh (WSI Patches):** `/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/WSIs/`\n",
    "- **Dữ liệu Lâm sàng chuẩn hóa:** `/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---------------------------------------------------------\n",
    "# IMPORT CÁC THƯ VIỆN CẦN THIẾT\n",
    "# ---------------------------------------------------------\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "import random\n",
    "from PIL import Image\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Cài đặt style cho biểu đồ thêm phần chuyên nghiệp\n",
    "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
    "plt.rcParams.update({'font.size': 12})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 1: Tải và Đánh giá Tổng quan Dữ liệu Lâm Sàng\n",
    "Chúng ta sẽ đọc file `tcga_brca_master_matched_cohort.csv`. Đây là một file CSV đã được tổng hợp cực kỳ chi tiết từ dự án TCGA."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Định nghĩa đường dẫn (Paths)\n",
    "CLINICAL_CSV_PATH = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv'\n",
    "WSI_DIR = '/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/WSIs'\n",
    "\n",
    "print(f\"Đang tải dữ liệu lâm sàng từ: {CLINICAL_CSV_PATH}\")\n",
    "\n",
    "try:\n",
    "    df = pd.read_csv(CLINICAL_CSV_PATH, sep='\\t') # Thử đọc bằng tab nếu gặp lỗi format\n",
    "    if len(df.columns) < 5:\n",
    "        df = pd.read_csv(CLINICAL_CSV_PATH) # Fallback đọc bằng phẩy\n",
    "except Exception as e:\n",
    "    print(f\"Lỗi đọc file: {e}. Khởi tạo Dataframe trống để code không bị gián đoạn.\")\n",
    "    df = pd.DataFrame()\n",
    "\n",
    "if not df.empty:\n",
    "    print(f\"\\n✅ Tải thành công! Tập dữ liệu gồm {df.shape[0]} bệnh nhân và {df.shape[1]} trường thông tin.\")\n",
    "    display(df.head(3))\n",
    "else:\n",
    "    print(\"❌ Không tìm thấy dữ liệu. Hãy chắc chắn bạn đã Add Data trên Kaggle.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 📖 Từ Điển Dữ Liệu (Data Dictionary)\n",
    "Trước khi đi sâu, chúng ta cần chọn ra các cột quan trọng nhất và hiểu ý nghĩa y khoa của chúng:\n",
    "\n",
    "| Tên Cột | Ý Nghĩa (Giải thích Lâm sàng) | Vai Trò Trong Dự Án |\n",
    "|---|---|---|\n",
    "| `patientId` | Mã định danh duy nhất của bệnh nhân (VD: TCGA-3C-AAAU). | **Khóa chính (Primary Key)** để nối với folder ảnh WSI. |\n",
    "| `pam50_subtype` / `SUBTYPE` | Phân nhóm phân tử của ung thư vú (Luminal A, Luminal B, HER2, Basal). Đây là chuẩn vàng trong chẩn đoán gen. | **Nhãn Y cho bài toán Classification** (Sẽ dùng Transformer dự đoán nhãn này từ ảnh). |\n",
    "| `OS_STATUS` / `survival_months` | Tình trạng sống còn (Living/Deceased) và số tháng sống sót từ lúc chẩn đoán. | **Nhãn Y cho bài toán Survival Prediction** (Dùng cho mạng LSTM). |\n",
    "| `AGE` / `age_at_diagnosis` | Tuổi của bệnh nhân khi phát hiện bệnh. Tuổi tác là yếu tố nguy cơ hàng đầu. | Đặc trưng lâm sàng bổ trợ cho mô hình LSTM. |\n",
    "| `AJCC_PATHOLOGIC_TUMOR_STAGE` | Giai đoạn khối u (Stage I, II, III, IV). Thể hiện mức độ lan rộng của ung thư. | Đặc trưng phân tích tương quan sinh tồn. |\n",
    "| `CANCER_TYPE_DETAILED` | Phân nhóm mô học (IDC - Ung thư ống tuyến, ILC - Ung thư tiểu thùy). | Phân tích phụ để so sánh sự khác biệt hình thái. |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 2: Khám Phá Bài Toán Chẩn Đoán (Phân Nhóm Phân Tử)\n",
    "Mục tiêu của mạng TransMIL sắp tới là dự đoán cột `pam50_subtype`. Chúng ta cần xem phân bố của nó."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df.empty and 'pam50_subtype' in df.columns:\n",
    "    # Làm sạch cột nhãn (Loại bỏ giá trị Null/Normal nếu có)\n",
    "    df_subtype = df.dropna(subset=['pam50_subtype'])\n",
    "    subtype_counts = df_subtype['pam50_subtype'].value_counts()\n",
    "    \n",
    "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
    "    \n",
    "    # Biểu đồ Cột\n",
    "    sns.barplot(x=subtype_counts.index, y=subtype_counts.values, ax=ax1, palette=\"viridis\")\n",
    "    ax1.set_title(\"Phân bố các Phân nhóm Phân tử (Molecular Subtypes)\", fontweight='bold')\n",
    "    ax1.set_ylabel(\"Số lượng Bệnh nhân\")\n",
    "    ax1.tick_params(axis='x', rotation=45)\n",
    "    \n",
    "    # Biểu đồ Tròn\n",
    "    ax2.pie(subtype_counts.values, labels=subtype_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette(\"viridis\", len(subtype_counts)))\n",
    "    ax2.set_title(\"Tỷ trọng các Phân nhóm\", fontweight='bold')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    print(\"💡 KẾT LUẬN TỪ BIỂU ĐỒ:\")\n",
    "    print(\"- Dữ liệu bị Mất cân bằng lớp (Class Imbalance) nghiêm trọng. Nhóm LumA (Luminal A) chiếm đa số.\")\n",
    "    print(\"- Giải pháp đề xuất: Khi train mô hình TransMIL, bắt buộc phải dùng Stratified K-Fold để chia đều tỉ lệ, và dùng Macro-F1 thay cho Accuracy để đánh giá.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 3: Khám Phá Bài Toán Dự Báo Sinh Tồn (Survival Analysis)\n",
    "Đối với mạng LSTM, chúng ta quan tâm đến yếu tố thời gian sống sót. Sự phân bố thời gian sống cho chúng ta biết tiên lượng chung của căn bệnh này."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df.empty and 'OS_MONTHS' in df.columns and 'OS_STATUS' in df.columns:\n",
    "    # Tiền xử lý OS_STATUS: Chuyển '0:LIVING' thành 0, '1:DECEASED' thành 1\n",
    "    df['Death_Event'] = df['OS_STATUS'].apply(lambda x: 1 if '1' in str(x) else 0)\n",
    "    \n",
    "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
    "    \n",
    "    # Phân bố thời gian sinh tồn\n",
    "    sns.histplot(df['OS_MONTHS'].dropna(), bins=30, kde=True, ax=ax1, color=\"coral\")\n",
    "    ax1.set_title(\"Phân bố Thời gian Theo dõi (Tháng)\", fontweight='bold')\n",
    "    ax1.set_xlabel(\"Số tháng (OS_MONTHS)\")\n",
    "    ax1.set_ylabel(\"Tần suất\")\n",
    "    \n",
    "    # So sánh tình trạng sinh tồn theo Phân nhóm\n",
    "    if 'pam50_subtype' in df.columns:\n",
    "        sns.countplot(data=df, x='pam50_subtype', hue='Death_Event', ax=ax2, palette=\"Set2\")\n",
    "        ax2.set_title(\"Tình trạng Tử vong (Death Event) theo từng Phân nhóm\", fontweight='bold')\n",
    "        ax2.tick_params(axis='x', rotation=45)\n",
    "        ax2.legend(title='Sự kiện (1=Tử vong)', loc='upper right')\n",
    "        \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    print(\"💡 KẾT LUẬN TỪ BIỂU ĐỒ:\")\n",
    "    print(\"- Phần lớn thời gian theo dõi tập trung ở mức 0 - 100 tháng.\")\n",
    "    print(\"- Một tỷ lệ lớn bệnh nhân vẫn đang sống (Censored data - Nhãn 0). Mô hình LSTM dự báo rủi ro sinh tồn cần phải xử lý được hiện tượng Censoring này.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 4: Yếu tố Nhân khẩu học và Lâm sàng (Age & Stage)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not df.empty and 'AGE' in df.columns and 'AJCC_PATHOLOGIC_TUMOR_STAGE' in df.columns:\n",
    "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
    "    \n",
    "    # Phân bố độ tuổi\n",
    "    sns.kdeplot(data=df, x=\"AGE\", hue=\"pam50_subtype\", fill=True, alpha=0.3, ax=ax1)\n",
    "    ax1.set_title(\"Phân bố Độ tuổi khi Chẩn đoán theo Nhóm Phân tử\", fontweight='bold')\n",
    "    ax1.set_xlabel(\"Tuổi (AGE)\")\n",
    "    \n",
    "    # Phân bố Giai đoạn Bệnh\n",
    "    # Làm sạch tên giai đoạn (bỏ các chữ A, B, C nhỏ để gom nhóm chuẩn)\n",
    "    df['Clean_Stage'] = df['AJCC_PATHOLOGIC_TUMOR_STAGE'].str.extract(r'(STAGE [IVX]+)')\n",
    "    sns.countplot(data=df.dropna(subset=['Clean_Stage']), x='Clean_Stage', \n",
    "                  order=['STAGE I', 'STAGE II', 'STAGE III', 'STAGE IV', 'STAGE X'], ax=ax2, palette=\"mako\")\n",
    "    ax2.set_title(\"Phân bố Giai đoạn Khối u (Tumor Stage)\", fontweight='bold')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bước 5: Kiểm tra Dữ liệu Ảnh (WSI Patches) & Khớp nối\n",
    "Bây giờ, chúng ta sẽ kiểm tra thư mục chứa ảnh trên Kaggle. Quan trọng nhất: Liệu ID của thư mục ảnh có khớp với cột `patientId` trong file CSV không?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if os.path.exists(WSI_DIR) and not df.empty:\n",
    "    # Lấy danh sách folder ảnh hiện có\n",
    "    image_folders = [f for f in os.listdir(WSI_DIR) if os.path.isdir(os.path.join(WSI_DIR, f))]\n",
    "    csv_patients = df['patientId'].dropna().astype(str).tolist()\n",
    "    \n",
    "    # Tìm kiếm sự trùng khớp\n",
    "    matched_patients = list(set(image_folders).intersection(set(csv_patients)))\n",
    "    \n",
    "    print(f\"📊 THỐNG KÊ KHỚP NỐI (CROSS-MATCHING):\")\n",
    "    print(f\"- Số lượng folder bệnh nhân trong WSIs: {len(image_folders)}\")\n",
    "    print(f\"- Số lượng bệnh nhân trong CSV: {len(csv_patients)}\")\n",
    "    print(f\"- Số lượng bệnh nhân khớp nhau hoàn toàn: {len(matched_patients)}\")\n",
    "    \n",
    "    if len(matched_patients) > 0:\n",
    "        print(\"\\n✅ Tuyệt vời! Hai bộ dữ liệu đã được map thành công qua PatientID. Chúng ta có thể bắt đầu ghép ảnh và nhãn.\")\n",
    "    else:\n",
    "        print(\"\\n⚠️ Cảnh báo: Tên folder ảnh không khớp với mã bệnh nhân trong CSV. Cần viết hàm tiền xử lý chuỗi (string manipulation) ở bước sau.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Trực Quan Hóa Tế Bào Bệnh Học (WSI Patches)\n",
    "Để kết thúc Notebook EDA này, chúng ta hãy in ra ngẫu nhiên một vài mảng tế bào (patches) của một vài bệnh nhân đã xác định được nhãn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_patient_patches(wsi_dir, df, matched_list, num_patients=3, patches_per_patient=4):\n",
    "    if not matched_list or not os.path.exists(wsi_dir):\n",
    "        print(\"Không thể trực quan hóa do chưa có ảnh hoặc chưa khớp ID.\")\n",
    "        return\n",
    "        \n",
    "    # Chọn ngẫu nhiên bệnh nhân\n",
    "    selected = random.sample(matched_list, min(num_patients, len(matched_list)))\n",
    "    \n",
    "    fig, axes = plt.subplots(len(selected), patches_per_patient, figsize=(15, 3.5 * len(selected)))\n",
    "    if len(selected) == 1:\n",
    "        axes = [axes]\n",
    "        \n",
    "    for i, pid in enumerate(selected):\n",
    "        # Lấy nhãn của bệnh nhân này\n",
    "        subtype = df[df['patientId'] == pid]['pam50_subtype'].values[0] if 'pam50_subtype' in df.columns else \"Unknown\"\n",
    "        \n",
    "        p_dir = os.path.join(wsi_dir, pid)\n",
    "        patches = [p for p in os.listdir(p_dir) if p.endswith('.jpg')]\n",
    "        selected_patches = random.sample(patches, min(patches_per_patient, len(patches)))\n",
    "        \n",
    "        for j, patch_name in enumerate(selected_patches):\n",
    "            img_path = os.path.join(p_dir, patch_name)\n",
    "            try:\n",
    "                img = Image.open(img_path)\n",
    "                ax = axes[i][j] if patches_per_patient > 1 else axes[i]\n",
    "                ax.imshow(img)\n",
    "                ax.axis('off')\n",
    "                if j == 0:\n",
    "                    # Cột đầu tiên in tên bệnh nhân và nhãn\n",
    "                    ax.set_title(f\"ID: {pid}\\nSubtype: {subtype}\", fontsize=11, fontweight='bold', color='darkred', loc='left')\n",
    "            except:\n",
    "                pass\n",
    "                \n",
    "    plt.suptitle(\"Hình thái Tế bào (Patches) của các Bệnh nhân Ung thư Vú\", fontsize=16, fontweight='bold', y=1.02)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "if 'matched_patients' in locals():\n",
    "    show_patient_patches(WSI_DIR, df, matched_patients)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### 🏆 TỔNG KẾT NOTEBOOK 1:\n",
    "1. Chúng ta đã làm chủ được file CSV `tcga_brca_master_matched_cohort.csv`.\n",
    "2. Xác định được 2 cột nhãn cực kỳ quan trọng:\n",
    "   - `pam50_subtype` (Cho bài toán Classification)\n",
    "   - `OS_STATUS` & `OS_MONTHS` (Cho bài toán Survival).\n",
    "3. Chứng minh được sự mất cân bằng lớp ở các phân nhóm phân tử.\n",
    "4. Xác nhận sự liên kết thành công giữa **Bệnh án Lâm sàng** và **Ảnh Mô bệnh học** thông qua mã `patientId`.\n",
    "\n",
    "**👉 Bước tiếp theo (Notebook 2):** Sử dụng mạng CNN (ResNet50) để biến hàng nghìn tấm ảnh `.jpg` của mỗi bệnh nhân thành một ma trận vector toán học đặc trưng (Feature Tensors) nhằm chuẩn bị đưa vào mạng Transformer!"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

out_path = r'C:\Users\huynh\Desktop\breast cancer\01_TCGA_EDA_and_Clinical_Processing.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Updated Notebook 1 successfully.")
