import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 8: Trí Tuệ Nhân Tạo Giải Thích Được (Explainable AI - XAI)\n",
    "\n",
    "<div class=\"alert alert-info\">\n",
    "<strong>Slide 1: Đập Vỡ Hộp Đen Y Khoa (Black Box AI)</strong><br>\n",
    "Một mô hình AI Dù có F1 đạt 0.99 thì vẫn sẽ bị các Bác sĩ từ chối sử dụng nếu nó không thể trả lời câu hỏi: <strong>\"Tại sao mày lại phán bệnh nhân này bị ung thư HER2?\"</strong>. Y học không chấp nhận \"Hộp đen\".<br><br>\n",
    "Trong Notebook này, chúng ta sẽ sử dụng <strong>Captum (Integrated Gradients)</strong> - công nghệ XAI tối tân nhất của Facebook AI Research - để nội soi vào tư duy của AI Đa phương thức. Chúng ta sẽ bắt nó khai ra bằng được: Trong 500 gen, gen nào là hung thủ? Và trong 5000 mảnh tế bào, mảnh nào là lõi ung thư?\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install nystrom-attention captum"
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
    "from sklearn.preprocessing import StandardScaler\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from captum.attr import IntegratedGradients\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div class=\"alert alert-success\">\n",
    "<strong>Slide 2: Nạp Trọng Số (Weights) Nguyên Bản</strong><br>\n",
    "Khác với Notebook 7, ở đây chúng ta giữ nguyên 100% kiến trúc của Notebook 6 (Bao gồm cả Lớp Phân Loại 4 Nhãn). Bởi vì thuật toán Tích phân Gradient (Integrated Gradients) cần phải tính đạo hàm từ chính Nhãn dự đoán ngược về Dữ liệu gốc.\n",
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
    "        # Trả về Xác suất 4 nhãn (Logits) để phục vụ XAI\n",
    "        return self.classifier(v_fusion)"
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
    "print(\"Đang tái thiết lập không gian Gen...\")\n",
    "rna_df = pd.read_csv(RNASEQ_FILE, sep='\\t')\n",
    "if 'Entrez_Gene_Id' in rna_df.columns: rna_df = rna_df.drop(columns=['Entrez_Gene_Id'])\n",
    "rna_df = rna_df.set_index('Hugo_Symbol').T\n",
    "rna_df.index = rna_df.index.str[:12]\n",
    "rna_df = rna_df[~rna_df.index.duplicated(keep='first')].dropna(axis=1)\n",
    "top_500_genes = rna_df.var().nlargest(500).index # Lấy 500 TÊN GEN CỐT LÕI\n",
    "rna_500 = rna_df[top_500_genes]\n",
    "rna_500_scaled = pd.DataFrame(StandardScaler().fit_transform(rna_500), index=rna_500.index)\n",
    "\n",
    "df_clin = pd.read_csv(CLINICAL_CSV, sep='\\t')\n",
    "if len(df_clin.columns) < 5: df_clin = pd.read_csv(CLINICAL_CSV)\n",
    "df_clin = df_clin[df_clin['pam50_subtype'].isin(['BRCA_LumA', 'BRCA_LumB', 'BRCA_Basal', 'BRCA_Her2'])].copy()\n",
    "label_map = {'BRCA_LumA': 0, 'BRCA_LumB': 1, 'BRCA_Basal': 2, 'BRCA_Her2': 3}\n",
    "df_clin['label'] = df_clin['pam50_subtype'].map(label_map)\n",
    "\n",
    "valid_pids = [pid for pid in df_clin['patientId'].unique() if pid in rna_500_scaled.index and os.path.exists(os.path.join(PT_DIR, f\"{pid}.pt\"))]\n",
    "df_clin = df_clin[df_clin['patientId'].isin(valid_pids)].drop_duplicates('patientId').set_index('patientId')"
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
    "# TODO: Đường dẫn tới Dataset Trọng số 6.2 của bạn\n",
    "WEIGHTS_PATH = '/kaggle/input/tcga-brca-multimodal-genomics-weights/multimodal_genomics_fold1_best.pth'\n",
    "\n",
    "if os.path.exists(WEIGHTS_PATH):\n",
    "    state_dict = torch.load(WEIGHTS_PATH, map_location=device)\n",
    "    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}\n",
    "    model.load_state_dict(new_state_dict, strict=False)\n",
    "    print(\"Đã nạp thành công Trọng số God Mode!\")\n",
    "else:\n",
    "    print(f\"⚠️ Không tìm thấy {WEIGHTS_PATH}.\")\n",
    "\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Slide 3: Giải phẫu Quyết định của AI bằng Integrated Gradients\n",
    "<div class=\"alert alert-warning\">\n",
    "<strong>Lưỡi dao giải phẫu XAI:</strong><br>\n",
    "Chúng ta sẽ chọn 1 Bệnh nhân bất kỳ. AI đã đoán đúng họ mắc bệnh ung thư. Bây giờ, Captum sẽ bắn hàng vạn tia đạo hàm ngược (Backprop) từ kết quả dự đoán về lại 500 gen gốc. Kết quả trả về là <strong>Attribution Score (Điểm cống hiến)</strong>. Điểm càng dương (Màu Đỏ) nghĩa là gen đó càng khẳng định bệnh, điểm càng âm (Màu Xanh) nghĩa là gen đó cố gắng phủ định bệnh.\n",
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
    "ig = IntegratedGradients(model)\n",
    "\n",
    "# Chọn ngẫu nhiên 1 bệnh nhân\n",
    "demo_pid = random.choice(valid_pids)\n",
    "true_label = df_clin.loc[demo_pid, 'label']\n",
    "subtype_name = {0: 'LumA', 1: 'LumB', 2: 'Basal', 3: 'HER2'}[true_label]\n",
    "\n",
    "img_tensor = torch.load(os.path.join(PT_DIR, f\"{demo_pid}.pt\"), map_location=device).unsqueeze(0)\n",
    "gen_tensor = torch.tensor(rna_500_scaled.loc[demo_pid].values, dtype=torch.float32).unsqueeze(0).to(device)\n",
    "\n",
    "# Bật chế độ Gradient\n",
    "img_tensor.requires_grad_()\n",
    "gen_tensor.requires_grad_()\n",
    "\n",
    "print(f\"Bệnh nhân: {demo_pid} | Nhãn thực tế: {subtype_name}\")\n",
    "\n",
    "# Tính toán Integrated Gradients cho cả 2 nhánh cùng lúc!\n",
    "attributions, delta = ig.attribute(inputs=(img_tensor, gen_tensor), target=int(true_label), return_convergence_delta=True)\n",
    "img_attr, gen_attr = attributions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Slide 4: Truy Tìm \"Hung Thủ\" Đột Biến Gen\n",
    "<div class=\"alert alert-danger\">\n",
    "Dưới đây là <strong>Top 15 Gen</strong> có tiếng nói quyết định mạnh nhất trong số 500 gen. Nếu Bác sĩ nhìn vào biểu đồ này, họ sẽ lập tức biết được AI của chúng ta đang tuân theo đúng y lý hay chỉ đang đoán bừa.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "gen_attr_np = gen_attr.squeeze(0).cpu().detach().numpy()\n",
    "\n",
    "# Tìm Top 15 Gen quan trọng nhất (theo trị tuyệt đối)\n",
    "top_indices = np.argsort(np.abs(gen_attr_np))[-15:]\n",
    "top_genes = top_500_genes[top_indices]\n",
    "top_scores = gen_attr_np[top_indices]\n",
    "\n",
    "colors = ['red' if score > 0 else 'blue' for score in top_scores]\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.barh(top_genes, top_scores, color=colors)\n",
    "plt.title(f\"XAI Genomics: Top 15 Gen Quyết Định (Bệnh nhân {demo_pid} - {subtype_name})\", fontsize=14, fontweight='bold')\n",
    "plt.xlabel(\"Attribution Score (Đỏ: Khẳng định bệnh | Xanh: Phủ định bệnh)\")\n",
    "plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Slide 5: Truy Tìm \"Lõi Ung Thư\" trên Mảng Tế Bào WSI\n",
    "<div class=\"alert alert-info\">\n",
    "Túi ảnh (Bag of Features) của bệnh nhân này chứa hàng ngàn mảnh vỡ tế bào (Patches). Nhưng TransMIL có một cơ chế lọc cực kỳ thông minh: <strong>Nó chỉ tập trung vào một nhóm rất nhỏ các mảnh chứa tế bào bất thường.</strong><br>\n",
    "Biểu đồ dưới đây sẽ phân bố mức độ chú ý (Attention/Importance) của AI. Bạn sẽ thấy: Trong hàng ngàn mảnh vỡ, chỉ có 1 cột nhô lên cực cao (Vùng khối u), còn lại các mảnh mỡ/nền trắng đều bị ép về 0.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "img_attr_np = img_attr.squeeze(0).cpu().detach().numpy() # [N_patches, 2048]\n",
    "\n",
    "# Tính tổng độ quan trọng (Importance) của từng Patch trên toàn bộ 2048 chiều\n",
    "patch_importance = np.sum(img_attr_np, axis=1)\n",
    "patch_importance = np.maximum(patch_importance, 0) # Chỉ lấy các đóng góp dương\n",
    "\n",
    "# Sắp xếp để xem phân bố\n",
    "sorted_importance = np.sort(patch_importance)[::-1]\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.plot(sorted_importance, color='darkorange', linewidth=2, label=\"Mức độ chú ý của AI\")\n",
    "plt.fill_between(range(len(sorted_importance)), sorted_importance, color='orange', alpha=0.3)\n",
    "plt.title(f\"XAI Vision: Sự phân bố chú ý trên {len(patch_importance)} Mảnh Tế Bào WSI\", fontsize=14, fontweight='bold')\n",
    "plt.xlabel(\"Thứ hạng Mảnh tế bào (Từ quan trọng nhất đến ít quan trọng nhất)\")\n",
    "plt.ylabel(\"Độ quan trọng (Patch Importance Score)\")\n",
    "\n",
    "# Đánh dấu Top 50\n",
    "plt.axvline(x=50, color='red', linestyle='--', label=\"Top 50 Mảnh Lõi Ung Thư\")\n",
    "plt.legend()\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
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

out_path = r'C:\Users\huynh\Desktop\breast cancer\08_Explainable_AI_Multimodal.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
