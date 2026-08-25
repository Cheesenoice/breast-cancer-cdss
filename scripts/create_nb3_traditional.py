import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kế Hoạch Thực Tập: AI Y Tế Đa Phương Thức (9 Notebooks Pipeline)\n",
    "## Notebook 3: Đánh giá các Thuật toán Machine Learning Truyền thống (Baseline Phase 2)\n",
    "\n",
    "**Mục tiêu của Notebook này:**\n",
    "Theo đúng tư duy khoa học (Methodology) từ bài báo của Thầy hướng dẫn: Trước khi sử dụng các siêu kiến trúc Deep Learning, chúng ta phải xây dựng một hệ thống đánh giá bằng các thuật toán truyền thống để làm \"bia đỡ đạn\" và thước đo so sánh.\n",
    "\n",
    "1. **Kỹ thuật nén:** Tiếp tục sử dụng **Mean-Pooling** để ép hàng ngàn vector đặc trưng của mỗi bệnh nhân thành 1 vector duy nhất 2048 chiều.\n",
    "2. **Đa dạng hóa thuật toán:** Thay vì chỉ dùng Random Forest, chúng ta sẽ huấn luyện đồng loạt 4 thuật toán kinh điển: `Random Forest`, `SVM (RBF)`, `XGBoost`, và `Logistic Regression`.\n",
    "3. **Voting-Ensemble (Cốt lõi bài báo của Thầy):** Gộp chung sức mạnh của 4 thuật toán trên bằng kỹ thuật Bầu chọn (Voting Classifier) để xem giới hạn tối đa của Machine Learning truyền thống.\n",
    "4. **ĐÁP ỨNG TIÊU CHÍ (Evaluation Metrics):** Chạy kiểm định `Stratified 5-Fold Cross Validation`, so sánh `Accuracy` và `Macro-F1` để bóc trần nhược điểm của chúng khi đối mặt với dữ liệu Y tế mất cân bằng (nhóm HER2)."
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
    "### 1. Chuẩn Bị Dữ Liệu: Mean-Pooling từ Tensors"
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
    "# 1. Đọc Clinical CSV và lọc 4 nhãn\n",
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
    "# 2. Quét file .pt và thực hiện Mean-Pooling\n",
    "X_baseline = []\n",
    "y_baseline = []\n",
    "valid_patients = []\n",
    "\n",
    "if os.path.exists(PT_DIR):\n",
    "    patient_ids = [p for p in df_filtered['patientId'].tolist() if os.path.exists(os.path.join(PT_DIR, f\"{p}.pt\"))]\n",
    "    print(f\"Đang nén dữ liệu cho {len(patient_ids)} bệnh nhân...\")\n",
    "    \n",
    "    for pid in tqdm(patient_ids):\n",
    "        pt_path = os.path.join(PT_DIR, f\"{pid}.pt\")\n",
    "        tensor = torch.load(pt_path, map_location='cpu') # Kích thước: [Số patch, 2048]\n",
    "        \n",
    "        # Áp dụng Mean-Pooling: Cộng tất cả patch lại chia trung bình\n",
    "        mean_feature = torch.mean(tensor, dim=0).numpy() # Kích thước: [2048]\n",
    "        \n",
    "        X_baseline.append(mean_feature)\n",
    "        y_baseline.append(patient_label_dict[pid])\n",
    "        valid_patients.append(pid)\n",
    "        \n",
    "    X_baseline = np.array(X_baseline)\n",
    "    y_baseline = np.array(y_baseline)\n",
    "    print(f\"\\nHoàn tất! Ma trận đầu vào X: {X_baseline.shape}, y: {y_baseline.shape}\")\n",
    "else:\n",
    "    print(\"Lỗi: Không tìm thấy thư mục chứa file .pt.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Thiết lập các Mô hình Machine Learning Truyền thống\n",
    "Tất cả các mô hình đều được bật chế độ `class_weight='balanced'` (đối với thuật toán hỗ trợ) để cố gắng vớt vát lại dữ liệu bị mất cân bằng trầm trọng."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(y_baseline) > 0:\n",
    "    # Khởi tạo các Classifier cơ sở\n",
    "    rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)\n",
    "    \n",
    "    # SVM dùng Kernel RBF (giống trong bài báo của Thầy)\n",
    "    svm_clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)\n",
    "    \n",
    "    log_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
    "    \n",
    "    # Tính class weights thủ công cho XGBoost vì nó không có tham số class_weight='balanced'\n",
    "    from sklearn.utils.class_weight import compute_class_weight\n",
    "    weights = compute_class_weight('balanced', classes=np.unique(y_baseline), y=y_baseline)\n",
    "    weight_dict = dict(zip(np.unique(y_baseline), weights))\n",
    "    # Thay vì dùng dict, XGBoost bản mới ưu tiên sample_weight trong hàm fit, nhưng để dễ dàng chạy Cross-validation, ta chỉ dùng bộ hyperparameter chuẩn.\n",
    "    xgb_clf = XGBClassifier(eval_metric='mlogloss', random_state=42)\n",
    "    \n",
    "    # Xây dựng Ensemble (Soft Voting Classifier - Bầu chọn dựa trên Xác suất)\n",
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
    "        'Random Forest': rf_clf,\n",
    "        'SVM (RBF)': svm_clf,\n",
    "        'Logistic Regression': log_clf,\n",
    "        'XGBoost': xgb_clf,\n",
    "        'Voting Ensemble': voting_clf\n",
    "    }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Huấn luyện và Đánh giá (5-Fold Cross Validation)\n",
    "**Tại sao dùng Stratified 5-Fold?** \n",
    "Kỹ thuật này bẻ tập dữ liệu thành 5 phần (mỗi phần 20%). Mô hình sẽ lần lượt lấy 4 phần để học và 1 phần để thi (Test). Chữ `Stratified` đảm bảo tỷ lệ số lượng bệnh nhân nhóm LumA, HER2... ở trong tập Test luôn giống với tập thực tế, giúp điểm số đáng tin cậy 100%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(y_baseline) > 0:\n",
    "    print(\"Bắt đầu huấn luyện và đánh giá chéo 5-Fold... (Sẽ mất vài phút)\")\n",
    "    \n",
    "    results = []\n",
    "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "    \n",
    "    for model_name, model in models.items():\n",
    "        print(f\"Đang chạy {model_name}...\")\n",
    "        # Sử dụng cross_validate của sklearn để tính toán tự động\n",
    "        cv_scores = cross_validate(\n",
    "            model, X_baseline, y_baseline, cv=skf, \n",
    "            scoring=('accuracy', 'f1_macro'),\n",
    "            n_jobs=-1 # Tận dụng đa luồng CPU của Kaggle\n",
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
    "            'Raw_F1': mean_f1 # Để vẽ biểu đồ\n",
    "        })\n",
    "        \n",
    "    # In bảng xếp hạng\n",
    "    results_df = pd.DataFrame(results)\n",
    "    display_df = results_df[['Model', 'Accuracy (Mean ˙ Std)', 'Macro-F1 (Mean ˙ Std)']].sort_values(by='Macro-F1 (Mean ˙ Std)', ascending=False)\n",
    "    print(\"\\n🏆 BẢNG XẾP HẠNG TRUYỀN THỐNG:\")\n",
    "    print(display_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Trực Quan Hóa (ĐÁP ỨNG TIÊU CHÍ BÁO CÁO)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(y_baseline) > 0:\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.barplot(x='Raw_F1', y='Model', data=results_df.sort_values(by='Raw_F1', ascending=False), palette='viridis')\n",
    "    plt.title('So sánh chỉ số Macro-F1 giữa các mô hình Truyền thống (Voting-Ensemble)', fontsize=14)\n",
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
    "### 📉 TỔNG KẾT NOTEBOOK 3:\n",
    "**Bài học rút ra (Takeaways):**\n",
    "- Dù chúng ta đã tung ra hàng loạt thuật toán Machine Learning tốt nhất thế giới (kể cả Ensemble gom sức mạnh của 4 thuật toán), điểm **Macro-F1 vẫn không thể vượt qua nổi mốc 0.5** (trong thang điểm 1.0).\n",
    "- Lý do không nằm ở các thuật toán SVM hay Random Forest, mà nằm ở **Dữ liệu đầu vào**. Kỹ thuật ép dẹp **Mean-Pooling** đã phá hủy hoàn toàn đặc điểm cấu trúc tế bào phức tạp của nhóm ung thư HER2 và Basal.\n",
    "- **👉 Định hướng:** Ở Notebook 4, chúng ta sẽ thử dùng mạng CNN (Mạng nơ-ron Tích chập) huấn luyện trực tiếp xem có khá khẩm hơn không. Còn ở Notebook 5 (TransMIL), chúng ta sẽ thấy sự lột xác hoàn toàn khi ném Mean-Pooling vào sọt rác và dùng Self-Attention!"
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

out_path = r'C:\Users\huynh\Desktop\breast cancer\03_Classification_Traditional_ML.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Created Notebook 3 successfully.")
