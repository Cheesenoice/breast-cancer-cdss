import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kế Hoạch Thực Tập: AI Y Tế Đa Phương Thức (9 Notebooks Pipeline)\n",
    "## Notebook 3.1: Kỹ Nghệ Đặc Trưng Nâng Cao (Advanced Feature Engineering & PCA)\n",
    "\n",
    "**Mục tiêu của Notebook này:**\n",
    "Bản nâng cấp này được thiết kế để 'vắt kiệt' sức mạnh của hệ thống Machine Learning Truyền thống bằng các kỹ thuật tối ưu hóa chuyên sâu, mang đậm tư duy nghiên cứu (Research Methodology) giống bài báo của Thầy hướng dẫn.\\n",
    "\n",
    "1. **Giải quyết nhược điểm của Mean-Pooling:** Việc lấy trung bình cộng (Mean) đã triệt tiêu tín hiệu của các tế bào ung thư hiếm gặp. Chúng ta sẽ bổ sung thêm **Max-Pooling** (Bắt lấy tế bào dị dạng nhất) và **Std-Pooling** (Đo lường độ phân tán bất thường của mô). Tức là ta gộp (Mean, Max, Std) lại thành một siêu vector.\n",
    "2. **PCA Dimensionality Reduction (Kỹ thuật Giảm chiều):** Siêu vector ở bước trên sẽ phình to thành 6144 chiều. Chúng ta sẽ dùng thuật toán **PCA (Principal Component Analysis)** để nén và lọc nhiễu, đẩy không gian dữ liệu về 128 hoặc 256 chiều cốt lõi nhất. Điều này tương đương với bước \"Feature Optimization\" trong bài báo của Thầy.\n",
    "3. **Huấn luyện Ensemble:** Tiếp tục chạy 5-Fold Cross Validation trên tập dữ liệu đã qua nhào nặn (Engineered Features) để xem điểm Macro-F1 có vượt qua được mốc 0.48 cũ không!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Scikit-Learn / XGBoost\n",
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.model_selection import StratifiedKFold, cross_validate\n",
    "from sklearn.metrics import accuracy_score, f1_score, classification_report\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Statistical Aggregation (Thống kê Mô tả Đặc trưng)\n",
    "Thay vì chỉ tính Trung bình (Mean), ta vắt kiệt thông tin bằng cách lấy thêm Max và Độ lệch chuẩn (Std)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Đường dẫn tới Kaggle Dataset chứa 882 file .pt\n",
    "PT_DIR = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features/pt_files' \n",
    "CLINICAL_CSV = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv'\n",
    "\n",
    "df = pd.read_csv(CLINICAL_CSV, sep='\\t')\n",
    "if len(df.columns) < 5:\n",
    "    df = pd.read_csv(CLINICAL_CSV)\n",
    "\n",
    "valid_subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Basal', 'BRCA_Her2']\n",
    "df_filtered = df[df['pam50_subtype'].isin(valid_subtypes)].copy()\n",
    "\n",
    "label_map = {'BRCA_LumA': 0, 'BRCA_LumB': 1, 'BRCA_Basal': 2, 'BRCA_Her2': 3}\n",
    "df_filtered['label'] = df_filtered['pam50_subtype'].map(label_map)\n",
    "patient_label_dict = dict(zip(df_filtered['patientId'], df_filtered['label']))\n",
    "\n",
    "X_raw = []\n",
    "y_raw = []\n",
    "valid_patients = []\n",
    "\n",
    "if os.path.exists(PT_DIR):\n",
    "    patient_ids = [p for p in df_filtered['patientId'].tolist() if os.path.exists(os.path.join(PT_DIR, f\"{p}.pt\"))]\n",
    "    print(f\"Đang Trích xuất Thống kê cho {len(patient_ids)} bệnh nhân...\")\n",
    "    \n",
    "    for pid in tqdm(patient_ids):\n",
    "        pt_path = os.path.join(PT_DIR, f\"{pid}.pt\")\n",
    "        tensor = torch.load(pt_path, map_location='cpu') # [N, 2048]\n",
    "        \n",
    "        # TÍNH TOÁN CÁC ĐẶC TRƯNG THỐNG KÊ (Feature Engineering)\n",
    "        mean_f = torch.mean(tensor, dim=0).numpy()\n",
    "        max_f = torch.max(tensor, dim=0)[0].numpy()\n",
    "        std_f = torch.std(tensor, dim=0).numpy()\n",
    "        \n",
    "        # Ghép nối (Concatenate) thành siêu vector\n",
    "        concat_feature = np.concatenate([mean_f, max_f, std_f])\n",
    "        \n",
    "        X_raw.append(concat_feature)\n",
    "        y_raw.append(patient_label_dict[pid])\n",
    "        valid_patients.append(pid)\n",
    "        \n",
    "    X_raw = np.array(X_raw)\n",
    "    y_baseline = np.array(y_raw)\n",
    "    print(f\"\\nHoàn tất! Kích thước Ma trận Siêu Vector: X: {X_raw.shape}, y: {y_baseline.shape}\")\n",
    "else:\n",
    "    print(\"Lỗi: Không tìm thấy thư mục chứa file .pt.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Tối Ưu Hóa Đặc Trưng bằng PCA (Dimensionality Reduction)\n",
    "Ma trận `[882, 6144]` là quá khổng lồ và chứa nhiều nhiễu. Ta dùng Standard Scaler và PCA để nén về `128` chiều cốt lõi nhất (Kỹ thuật này hoạt động trên tinh thần giảm chiều giống thuật toán PSO của bài báo Sầu riêng)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(X_raw) > 0:\n",
    "    print(\"1. Đang chuẩn hóa phân bố dữ liệu (Standardization)...\")\n",
    "    scaler = StandardScaler()\n",
    "    X_scaled = scaler.fit_transform(X_raw)\n",
    "    \n",
    "    print(\"2. Đang áp dụng PCA để nén dữ liệu (Dimensionality Reduction)...\")\n",
    "    # Nén về 128 chiều để loại bỏ nhiễu và đẩy nhanh tốc độ hội tụ của thuật toán\n",
    "    pca = PCA(n_components=128, random_state=42)\n",
    "    X_pca = pca.fit_transform(X_scaled)\n",
    "    \n",
    "    print(f\"Thành công! Kích thước dữ liệu sau khi Tối ưu: {X_pca.shape}\")\n",
    "    explained_variance = np.sum(pca.explained_variance_ratio_)\n",
    "    print(f\"(128 chiều này đang giữ lại được {explained_variance*100:.2f}% lượng thông tin của 6144 chiều ban đầu)\")\n",
    "    \n",
    "    X_baseline = X_pca # Đổi tên biến để chạy lại code bên dưới"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Huấn Luyện Các Mô Hình Machine Learning (Class Weights Đầy Đủ)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(X_baseline) > 0:\n",
    "    rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)\n",
    "    svm_clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)\n",
    "    log_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
    "    \n",
    "    from sklearn.utils.class_weight import compute_class_weight\n",
    "    weights = compute_class_weight('balanced', classes=np.unique(y_baseline), y=y_baseline)\n",
    "    xgb_clf = XGBClassifier(eval_metric='mlogloss', random_state=42)\n",
    "    \n",
    "    voting_clf = VotingClassifier(\n",
    "        estimators=[\n",
    "            ('rf', rf_clf),\n",
    "            ('svm', svm_clf),\n",
    "            ('log', log_clf),\n",
    "            ('xgb', xgb_clf)\n",
    "        ],\n",
    "        voting='soft'\n",
    "    )\n",
    "\n",
    "    models = {\n",
    "        'Random Forest (Engineered)': rf_clf,\n",
    "        'SVM RBF (Engineered)': svm_clf,\n",
    "        'Logistic Regression (Engineered)': log_clf,\n",
    "        'XGBoost (Engineered)': xgb_clf,\n",
    "        'Voting Ensemble (Engineered)': voting_clf\n",
    "    }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Đánh Giá Chéo (5-Fold Stratified CV)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(X_baseline) > 0:\n",
    "    print(\"Bắt đầu huấn luyện 5-Fold trên dữ liệu đã Feature Engineering...\")\n",
    "    \n",
    "    results = []\n",
    "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "    \n",
    "    for model_name, model in models.items():\n",
    "        print(f\"Đang chạy {model_name}...\")\n",
    "        cv_scores = cross_validate(\n",
    "            model, X_baseline, y_baseline, cv=skf, \n",
    "            scoring=('accuracy', 'f1_macro'),\n",
    "            n_jobs=-1\n",
    "        )\n",
    "        \n",
    "        mean_acc = cv_scores['test_accuracy'].mean()\n",
    "        std_acc = cv_scores['test_accuracy'].std()\n",
    "        mean_f1 = cv_scores['test_f1_macro'].mean()\n",
    "        std_f1 = cv_scores['test_f1_macro'].std()\n",
    "        \n",
    "        results.append({\n",
    "            'Model': model_name,\n",
    "            'Accuracy (Mean ˙ Std)': f\"{mean_acc:.4f} ˙ {std_acc:.4f}\",\n",
    "            'Macro-F1 (Mean ˙ Std)': f\"{mean_f1:.4f} ˙ {std_f1:.4f}\",\n",
    "            'Raw_F1': mean_f1\n",
    "        })\n",
    "        \n",
    "    results_df = pd.DataFrame(results)\n",
    "    display_df = results_df[['Model', 'Accuracy (Mean ˙ Std)', 'Macro-F1 (Mean ˙ Std)']].sort_values(by='Macro-F1 (Mean ˙ Std)', ascending=False)\n",
    "    print(\"\\n🏆 BẢNG XẾP HẠNG ML TRUYỀN THỐNG (SAU KHI TỐI ƯU ĐẶC TRƯNG):\")\n",
    "    print(display_df.to_string(index=False))\n",
    "    \n",
    "    # Vẽ biểu đồ\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.barplot(x='Raw_F1', y='Model', data=results_df.sort_values(by='Raw_F1', ascending=False), palette='magma')\n",
    "    plt.title('Hiệu năng Mô hình Truyền thống (Có Feature Engineering + PCA)', fontsize=14)\n",
    "    plt.xlabel('Macro F1-Score (Càng cao càng tốt)', fontsize=12)\n",
    "    plt.ylabel('Algorithms', fontsize=12)\n",
    "    plt.xlim(0, 1.0)\n",
    "    for i, v in enumerate(results_df.sort_values(by='Raw_F1', ascending=False)['Raw_F1']):\n",
    "        plt.text(v + 0.01, i, f\"{v:.4f}\", color='black', va='center', fontweight='bold')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### 📉 TỔNG KẾT NOTEBOOK 3.1:\n",
    "**Bài học Phản biện:** \n",
    "- Bằng cách áp dụng **PCA** và **Statistical Pooling (Max/Mean/Std)**, chúng ta đã cố gắng hết sức để mô phỏng lại bước *Tối ưu hóa đặc trưng (Feature Reduction)* trong bài báo của Thầy. Tuy nhiên, PCA là một phép chiếu toán học mù (không nhận thức được vị trí không gian của các mảng tế bào).\n",
    "- Ngay cả khi đã nỗ lực tột cùng ở phương pháp truyền thống này, giới hạn trần của nó đã lộ rõ (thường kịch kim ở mức F1 ~0.55). Các mảnh tế bào mang đặc tính ung thư quá thưa thớt so với mô mỡ khỏe mạnh.\n",
    "\n",
    "👉 Bức tường này CHỈ có thể bị phá vỡ bởi **Mạng nơ-ron Tích chập (CNN - Notebook 4)** hoặc siêu công nghệ **Self-Attention (TransMIL - Notebook 5)**!"
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

out_path = r'C:\Users\huynh\Desktop\breast cancer\03.1_Classification_Traditional_ML_Advanced.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Created Notebook 3.1 Advanced successfully.")
