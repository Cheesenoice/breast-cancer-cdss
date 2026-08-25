import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 7: Phân Tích Sinh Tồn (Survival Analysis) - Đỉnh Cao Lâm Sàng\n",
    "\n",
    "<div class=\"alert alert-info\">\n",
    "<strong>Slide 1: Từ Phân Loại đến Tiên Lượng Sinh Tồn</strong><br>\n",
    "Trong y học thực chứng, biết được \"Bệnh nhân mắc ung thư gì?\" chỉ là một nửa câu chuyện. Câu hỏi quan trọng hơn mà mọi bác sĩ đều phải đối mặt là: <strong>\"Bệnh nhân này sẽ sống được bao lâu?\"</strong> và <strong>\"Họ thuộc nhóm Nguy cơ cao hay Nguy cơ thấp?\"</strong>.<br><br>\n",
    "Trong Notebook này, chúng ta sẽ không huấn luyện AI lại từ đầu. Thay vào đó, chúng ta sẽ \"thỉnh\" (Load) siêu vector Đa phương thức (Ảnh + Gen) đã được tôi luyện từ Notebook 6, kết hợp với các thuật toán thống kê y khoa kinh điển để dự đoán Tuổi thọ của bệnh nhân!\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install nystrom-attention lifelines"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from nystrom_attention import NystromAttention\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import matplotlib.pyplot as plt\n",
    "from lifelines import CoxPHFitter, KaplanMeierFitter\n",
    "from lifelines.statistics import logrank_test\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-success\">\n",
    "<strong>Slide 2: Kiến trúc Trích xuất Đặc trưng (Feature Extraction)</strong><br>\n",
    "Chúng ta định nghĩa lại nguyên xi mạng lưới <code>Multimodal_GenomicsFusion</code> của Notebook 6. Tuy nhiên, lớp Classifier cuối cùng sẽ bị vô hiệu hóa. Chúng ta chỉ lấy output từ <strong>Lớp Hợp Nhất (Fusion Layer)</strong>, tạo ra một siêu vector 1024 chiều chứa đựng toàn bộ tri thức về Hình thái tế bào và Đột biến Gen của bệnh nhân.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class TransLayer(nn.Module):\n",
    "    def __init__(self, norm_layer=nn.LayerNorm, dim=512):\n",
    "        super().__init__()\n",
    "        self.norm = norm_layer(dim)\n",
    "        self.attn = NystromAttention(dim=dim, dim_head=dim//8, heads=8, num_landmarks=dim//2, pinv_iterations=6, residual=True, dropout=0.1)\n",
    "    def forward(self, x):\n",
    "        return x + self.attn(self.norm(x))\n",
    "\n",
    "class TransMIL_Vision_Extractor(nn.Module):\n",
    "    def __init__(self, input_dim=2048, out_dim=512):\n",
    "        super().__init__()\n",
    "        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())\n",
    "        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))\n",
    "        self.layer1 = TransLayer(dim=512)\n",
    "        self.layer2 = TransLayer(dim=512)\n",
    "        self.norm = nn.LayerNorm(512)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        h = self._fc1(x.float())\n",
    "        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1)\n",
    "        h = torch.cat((cls_tokens, h), dim=1)\n",
    "        h = self.layer1(h)\n",
    "        h = self.layer2(h)\n",
    "        return self.norm(h)[:,0]\n",
    "\n",
    "class Multimodal_GenomicsFusion(nn.Module):\n",
    "    def __init__(self, genomics_dim=500, num_classes=4):\n",
    "        super().__init__()\n",
    "        self.vision_net = TransMIL_Vision_Extractor()\n",
    "        self.genomics_net = nn.Sequential(\n",
    "            nn.Linear(genomics_dim, 256),\n",
    "            nn.LayerNorm(256),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.3),\n",
    "            nn.Linear(256, 512),\n",
    "            nn.ReLU()\n",
    "        )\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(512 + 512, 256),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.3),\n",
    "            nn.Linear(256, num_classes)\n",
    "        )\n",
    "        \n",
    "    def forward(self, img_feat, gen_feat):\n",
    "        v_img = self.vision_net(img_feat)\n",
    "        v_gen = self.genomics_net(gen_feat)\n",
    "        v_fusion = torch.cat((v_img, v_gen), dim=1)\n",
    "        # TRẢ VỀ SIÊU VECTOR 1024 CHIỀU, KHÔNG PHÂN LOẠI NỮA\n",
    "        return v_fusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "PT_DIR = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features/pt_files' \n",
    "RNASEQ_FILE = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/data_mrna_seq_v2_rsem.txt'\n",
    "CLINICAL_CSV = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv'\n",
    "\n",
    "# 1. Xử lý Dữ liệu Gen (Variance Threshold 500)\n",
    "print(\"Đang tải 500 Gen cốt lõi...\")\n",
    "rna_df = pd.read_csv(RNASEQ_FILE, sep='\\t')\n",
    "if 'Entrez_Gene_Id' in rna_df.columns: rna_df = rna_df.drop(columns=['Entrez_Gene_Id'])\n",
    "rna_df = rna_df.set_index('Hugo_Symbol').T\n",
    "rna_df.index = rna_df.index.str[:12]\n",
    "rna_df = rna_df[~rna_df.index.duplicated(keep='first')].dropna(axis=1)\n",
    "top_500_genes = rna_df.var().nlargest(500).index\n",
    "rna_500 = rna_df[top_500_genes]\n",
    "rna_500_scaled = pd.DataFrame(StandardScaler().fit_transform(rna_500), index=rna_500.index)\n",
    "\n",
    "# 2. Xử lý Dữ liệu Lâm sàng & Sinh tồn\n",
    "df_clin = pd.read_csv(CLINICAL_CSV, sep='\\t')\n",
    "if len(df_clin.columns) < 5: df_clin = pd.read_csv(CLINICAL_CSV)\n",
    "df_clin = df_clin.dropna(subset=['OS_MONTHS', 'OS_STATUS', 'AGE', 'AJCC_PATHOLOGIC_TUMOR_STAGE'])\n",
    "\n",
    "# Chuyển đổi Trạng thái Sống/Chết thành 1/0\n",
    "df_clin['OS_STATUS_NUM'] = df_clin['OS_STATUS'].apply(lambda x: 1 if 'DECEASED' in str(x) or '1:' in str(x) else 0)\n",
    "\n",
    "# Chuyển đổi Giai đoạn bệnh (Stage) thành số\n",
    "def parse_stage(s):\n",
    "    s = str(s).upper()\n",
    "    if 'IV' in s: return 4\n",
    "    if 'III' in s: return 3\n",
    "    if 'II' in s: return 2\n",
    "    if 'I' in s: return 1\n",
    "    return 1 # Default\n",
    "df_clin['STAGE_NUM'] = df_clin['AJCC_PATHOLOGIC_TUMOR_STAGE'].apply(parse_stage)\n",
    "\n",
    "valid_pids = [pid for pid in df_clin['patientId'].unique() if pid in rna_500_scaled.index and os.path.exists(os.path.join(PT_DIR, f\"{pid}.pt\"))]\n",
    "df_clin = df_clin[df_clin['patientId'].isin(valid_pids)].drop_duplicates('patientId').set_index('patientId')\n",
    "print(f\"Tìm thấy {len(valid_pids)} bệnh nhân hợp lệ cho Phân tích Sinh Tồn.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-warning\">\n",
    "<strong>Khởi chạy Inference (Siêu Tốc)</strong><br>\n",
    "Đoạn code dưới đây sẽ lấy Trọng số (Weights) từ Notebook 6 của bạn và ép qua 882 bệnh nhân chỉ trong vài chục giây để lấy ra \"Tinh hoa 1024 chiều\". Bạn nhớ chỉnh sửa đường dẫn file <code>.pth</code> bên dưới trỏ đúng vào Kaggle Dataset của bạn nhé!\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "model = Multimodal_GenomicsFusion().to(device)\n",
    "\n",
    "# TODO: Sửa đường dẫn này trỏ tới Kaggle Dataset chứa file weights của bạn!\n",
    "# Ví dụ: '/kaggle/input/tcga-brca-multimodal-genomics-weights/multimodal_genomics_fold1_best.pth'\n",
    "WEIGHTS_PATH = '/kaggle/input/TÊN_DATASET_CỦA_BẠN/multimodal_genomics_fold1_best.pth'\n",
    "\n",
    "if os.path.exists(WEIGHTS_PATH):\n",
    "    # Xử lý trường hợp mô hình lưu bằng DataParallel (có tiền tố 'module.')\n",
    "    state_dict = torch.load(WEIGHTS_PATH, map_location=device)\n",
    "    new_state_dict = {}\n",
    "    for k, v in state_dict.items():\n",
    "        name = k[7:] if k.startswith('module.') else k\n",
    "        new_state_dict[name] = v\n",
    "    model.load_state_dict(new_state_dict, strict=False)\n",
    "    print(\"Đã nạp thành công Trọng số Đa phương thức!\")\n",
    "else:\n",
    "    print(f\"⚠️ Không tìm thấy {WEIGHTS_PATH}. Sẽ dùng mô hình chưa train để minh họa (Chỉ để Test Code).\")\n",
    "\n",
    "model.eval()\n",
    "fusion_features = []\n",
    "\n",
    "print(\"Đang trích xuất Vector 1024 chiều...\")\n",
    "with torch.no_grad():\n",
    "    for pid in valid_pids:\n",
    "        img = torch.load(os.path.join(PT_DIR, f\"{pid}.pt\"), map_location=device).unsqueeze(0)\n",
    "        gen = torch.tensor(rna_500_scaled.loc[pid].values, dtype=torch.float32).unsqueeze(0).to(device)\n",
    "        v_fusion = model(img, gen)\n",
    "        fusion_features.append(v_fusion.cpu().numpy()[0])\n",
    "\n",
    "fusion_features = np.array(fusion_features) # [N, 1024]\n",
    "print(f\"Trích xuất xong! Kích thước ma trận: {fusion_features.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Slide 3: Cox Proportional Hazards & Giảm chiều PCA\n",
    "<div class=\"alert alert-danger\">\n",
    "<strong>Giải thuật CoxPH:</strong><br>\n",
    "Để thuật toán thống kê y khoa CoxPH không bị ngợp trước 1024 chiều dữ liệu AI, chúng ta sẽ dùng <strong>PCA</strong> để nén nó xuống còn 16 \"Thành phần chính\" (Principal Components). Những thành phần này, kết hợp cùng <strong>Tuổi (AGE)</strong> và <strong>Giai đoạn (STAGE)</strong>, sẽ được đưa vào mô hình để đo lường <em>Tỷ số Rủi ro (Hazard Ratio)</em>.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pca = PCA(n_components=16)\n",
    "fusion_pca = pca.fit_transform(fusion_features)\n",
    "\n",
    "surv_df = pd.DataFrame(fusion_pca, columns=[f'PC_{i}' for i in range(16)], index=valid_pids)\n",
    "surv_df['OS_MONTHS'] = df_clin['OS_MONTHS']\n",
    "surv_df['OS_STATUS'] = df_clin['OS_STATUS_NUM']\n",
    "surv_df['AGE'] = df_clin['AGE']\n",
    "surv_df['STAGE'] = df_clin['STAGE_NUM']\n",
    "\n",
    "# Fit mô hình CoxPH với L2 regularization để chống Overfitting\n",
    "cph = CoxPHFitter(penalizer=0.1)\n",
    "cph.fit(surv_df, duration_col='OS_MONTHS', event_col='OS_STATUS')\n",
    "\n",
    "print(f\"Concordance Index (C-Index): {cph.concordance_index_:.4f}\")\n",
    "print(\"(C-Index > 0.6 là tốt, > 0.7 là mô hình lâm sàng xuất sắc!)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Slide 4: Trực quan hóa Lâm sàng (Forest Plot & Kaplan-Meier)\n",
    "<div class=\"alert alert-info\">\n",
    "Đâu là tác nhân gây tử vong cao nhất? <strong>Biểu đồ Rừng (Forest Plot)</strong> dưới đây sẽ chỉ ra tầm quan trọng của Giai đoạn bệnh (STAGE), Tuổi (AGE) và các thành phần Trí tuệ Nhân tạo (PC_x).\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "cph.plot()\n",
    "plt.title(\"Biểu đồ Rừng (Forest Plot) - Đánh giá Tỷ số Rủi ro (Hazard Ratios)\", fontsize=14, fontweight='bold')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dự đoán Risk Score từ Cox Model\n",
    "surv_df['Risk_Score'] = cph.predict_partial_hazard(surv_df)\n",
    "median_risk = surv_df['Risk_Score'].median()\n",
    "\n",
    "high_risk = surv_df[surv_df['Risk_Score'] > median_risk]\n",
    "low_risk = surv_df[surv_df['Risk_Score'] <= median_risk]\n",
    "\n",
    "results = logrank_test(high_risk['OS_MONTHS'], low_risk['OS_MONTHS'], \n",
    "                       high_risk['OS_STATUS'], low_risk['OS_STATUS'])\n",
    "p_value = results.p_value\n",
    "\n",
    "plt.figure(figsize=(10, 7))\n",
    "kmf_high = KaplanMeierFitter()\n",
    "kmf_low = KaplanMeierFitter()\n",
    "\n",
    "kmf_high.fit(high_risk['OS_MONTHS'], event_observed=high_risk['OS_STATUS'], label='High Risk (Nguy cơ Cao)')\n",
    "kmf_low.fit(low_risk['OS_MONTHS'], event_observed=low_risk['OS_STATUS'], label='Low Risk (Nguy cơ Thấp)')\n",
    "\n",
    "kmf_high.plot_survival_function(color='red', lw=2)\n",
    "kmf_low.plot_survival_function(color='blue', lw=2)\n",
    "\n",
    "plt.title(\"Đường Cong Sinh Tồn Kaplan-Meier (Phân chia bởi AI Đa phương thức)\", fontsize=14, fontweight='bold')\n",
    "plt.xlabel(\"Thời gian (Tháng)\")\n",
    "plt.ylabel(\"Xác suất Sống sót\")\n",
    "plt.text(10, 0.2, f\"Log-Rank p-value: {p_value:.2e}\", fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))\n",
    "plt.grid(True, linestyle='--', alpha=0.6)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Slide 5: CASE STUDY DEMO (Cá nhân hóa Tiên lượng)\n",
    "<div class=\"alert alert-success\">\n",
    "Sức mạnh thực sự của Y tế Cá nhân hóa (Personalized Medicine) là đây! Hãy tưởng tượng bạn là Bác sĩ, bệnh nhân này vừa bước vào phòng khám của bạn. AI không chỉ chẩn đoán bệnh mà còn <strong>vẽ ra chính xác quỹ đạo sinh tồn</strong> của riêng họ trong 10 năm tới.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "\n",
    "# Bốc ngẫu nhiên 1 bệnh nhân\n",
    "demo_pid = random.choice(valid_pids)\n",
    "patient_row = surv_df.loc[[demo_pid]]\n",
    "\n",
    "real_months = patient_row['OS_MONTHS'].values[0]\n",
    "real_status = \"ĐÃ TỬ VONG\" if patient_row['OS_STATUS'].values[0] == 1 else \"CÒN SỐNG\"\n",
    "age = patient_row['AGE'].values[0]\n",
    "stage = patient_row['STAGE'].values[0]\n",
    "\n",
    "print(f\"==============================================\")\n",
    "print(f\" HỒ SƠ BỆNH NHÂN: {demo_pid}\")\n",
    "print(f\" Tuổi: {age} | Giai đoạn khối u: Stage {stage}\")\n",
    "print(f\" Sự thật lâm sàng: {real_status} (sau {real_months} tháng)\")\n",
    "print(f\"==============================================\")\n",
    "\n",
    "# Yêu cầu CoxPH vẽ đường cong tiên lượng riêng cho bệnh nhân này\n",
    "patient_survival = cph.predict_survival_function(patient_row)\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(patient_survival.index, patient_survival.iloc[:, 0], color='purple', lw=3, label='AI Tiên Lượng')\n",
    "\n",
    "if patient_row['OS_STATUS'].values[0] == 1:\n",
    "    plt.axvline(x=real_months, color='red', linestyle='--', label='Thời điểm Tử vong Thực tế')\n",
    "    plt.scatter(real_months, patient_survival.loc[real_months].values[0] if real_months in patient_survival.index else patient_survival.iloc[(patient_survival.index - real_months).abs().argsort()[:1]].values[0], color='red', s=100, zorder=5)\n",
    "else:\n",
    "    plt.axvline(x=real_months, color='green', linestyle='--', label='Lần Theo Dõi Cuối (Còn sống)')\n",
    "\n",
    "plt.title(f\"Quỹ Đạo Sinh Tồn Cá Nhân - Bệnh nhân {demo_pid}\", fontsize=14, fontweight='bold')\n",
    "plt.xlabel(\"Thời gian (Tháng)\")\n",
    "plt.ylabel(\"Xác suất Sống sót\")\n",
    "plt.legend()\n",
    "plt.grid(True, linestyle='--', alpha=0.6)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

out_path = r'C:\Users\huynh\Desktop\breast cancer\07_Survival_Analysis_Multimodal.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
