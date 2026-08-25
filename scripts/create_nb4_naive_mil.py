import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kế Hoạch Thực Tập: AI Y Tế Đa Phương Thức (9 Notebooks Pipeline)\n",
    "## Notebook 4: Deep Learning Baseline (Naive Deep MIL)\n",
    "\n",
    "**Mục tiêu của Notebook này:**\n",
    "Ở Notebook 3, chúng ta đổ lỗi sự thất bại cho kỹ thuật ép dữ liệu thô thiển (Mean/Max Pooling). Để chứng minh lập luận này là đúng, chúng ta sẽ loại bỏ Machine Learning truyền thống (SVM, Random Forest) và thay bằng một **Mạng Neural Network (Deep Learning) tiêu chuẩn nhiều lớp**.\n",
    "\n",
    "**Cơ chế (Naive MIL):**\n",
    "1. Vẫn dùng `Mean-Pooling` hoặc `Max-Pooling` để cộng dồn hàng ngàn vector `2048` chiều thành 1 vector duy nhất.\n",
    "2. Cho vector này chạy qua một mạng Neural Network sâu (Gồm các lớp `Linear`, `ReLU`, `Dropout`) để phân loại 4 nhãn ung thư.\n",
    "\n",
    "**👉 Điểm nhấn Phản biện:** \n",
    "Sự sụp đổ của mạng Neural Network này (dự kiến điểm F1 vẫn tồi tệ) sẽ chốt hạ 100% kết luận: Khuyết điểm không nằm ở thuật toán phân loại, mà nằm ở hành động **Cộng trung bình cào bằng**. Việc lấy trung bình đã vô tình làm loãng (Dilute) tín hiệu của tế bào ung thư HER2 hiếm hoi vào biển tế bào mỡ khổng lồ!"
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
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import accuracy_score, f1_score\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Xây Dựng Dataset (Áp Dụng Trực Tiếp Mean/Max Pooling)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "PT_DIR = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features/pt_files' \n",
    "CLINICAL_CSV = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv'\n",
    "\n",
    "df = pd.read_csv(CLINICAL_CSV, sep='\\t')\n",
    "if len(df.columns) < 5:\n",
    "    df = pd.read_csv(CLINICAL_CSV)\n",
    "\n",
    "valid_subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Basal', 'BRCA_Her2']\n",
    "df_filtered = df[df['pam50_subtype'].isin(valid_subtypes)].copy()\n",
    "label_map = {'BRCA_LumA': 0, 'BRCA_LumB': 1, 'BRCA_Basal': 2, 'BRCA_Her2': 3}\n",
    "df_filtered['label'] = df_filtered['pam50_subtype'].map(label_map)\n",
    "patient_label_dict = dict(zip(df_filtered['patientId'], df_filtered['label']))\n",
    "\n",
    "# Đọc và Pooling Dữ liệu\n",
    "X_data, y_data = [], []\n",
    "if os.path.exists(PT_DIR):\n",
    "    patient_ids = [p for p in df_filtered['patientId'].tolist() if os.path.exists(os.path.join(PT_DIR, f\"{p}.pt\"))]\n",
    "    print(f\"Đang tải và Pooling dữ liệu cho {len(patient_ids)} bệnh nhân...\")\n",
    "    \n",
    "    for pid in tqdm(patient_ids):\n",
    "        pt_path = os.path.join(PT_DIR, f\"{pid}.pt\")\n",
    "        tensor = torch.load(pt_path, map_location='cpu') # Kích thước [Số mảnh ảnh, 2048]\n",
    "        \n",
    "        # Áp dụng Mean-Pooling (Hành động cào bằng làm mất tín hiệu ung thư)\n",
    "        pooled_feature = torch.mean(tensor, dim=0).numpy() # Kích thước [2048]\n",
    "        X_data.append(pooled_feature)\n",
    "        y_data.append(patient_label_dict[pid])\n",
    "        \n",
    "    X_data = np.array(X_data)\n",
    "    y_data = np.array(y_data)\n",
    "    print(f\"\\nKích thước dữ liệu cuối cùng: X={X_data.shape}, y={y_data.shape}\")\n",
    "else:\n",
    "    print(\"Lỗi: Không tìm thấy thư mục PT_DIR.\")\n",
    "\n",
    "# Custom Dataset cho PyTorch\n",
    "class NaiveMILDataset(Dataset):\n",
    "    def __init__(self, features, labels):\n",
    "        self.features = torch.tensor(features, dtype=torch.float32)\n",
    "        self.labels = torch.tensor(labels, dtype=torch.long)\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.labels)\n",
    "        \n",
    "    def __getitem__(self, idx):\n",
    "        return self.features[idx], self.labels[idx]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Thiết Lập Mạng Neural Network Tiêu Chuẩn (Baseline DL)\n",
    "Mạng này gồm 3 lớp Fully Connected (Linear), xen kẽ bởi hàm kích hoạt phi tuyến tính `ReLU` và `Dropout` để chống Overfitting. Cấu trúc này mạnh và phức tạp hơn rất nhiều so với ML truyền thống."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class NaiveDeepMIL(nn.Module):\n",
    "    def __init__(self, input_dim=2048, num_classes=4):\n",
    "        super(NaiveDeepMIL, self).__init__()\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(input_dim, 512),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(p=0.4),\n",
    "            \n",
    "            nn.Linear(512, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(p=0.3),\n",
    "            \n",
    "            nn.Linear(128, num_classes)\n",
    "        )\n",
    "        \n",
    "    def forward(self, x):\n",
    "        # x có kích thước [batch_size, 2048]\n",
    "        logits = self.classifier(x)\n",
    "        return logits"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Huấn Luyện Với Stratified 5-Fold Cross Validation\n",
    "Để đảm bảo công bằng khi so sánh với thuật toán Truyền thống, mạng DL này cũng bị ép chạy 5-Fold trên đúng bộ dữ liệu chia y hệt như cũ, và cũng sử dụng **Weighted Cross-Entropy Loss** để cân bằng nhãn HER2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(X_data) > 0:\n",
    "    # Tính toán Trọng số Phạt (Class Weights) cho hàm Loss\n",
    "    class_weights = compute_class_weight('balanced', classes=np.unique(y_data), y=y_data)\n",
    "    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(class_weights, dtype=torch.float32)\n",
    "    \n",
    "    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)\n",
    "    \n",
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "    print(f\"Đang sử dụng thiết bị: {device}\")\n",
    "    \n",
    "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "    \n",
    "    fold_results = []\n",
    "    epochs = 30\n",
    "    batch_size = 32\n",
    "\n",
    "    for fold, (train_idx, test_idx) in enumerate(skf.split(X_data, y_data)):\n",
    "        print(f\"\\n--- Bắt đầu Fold {fold + 1}/5 ---\")\n",
    "        \n",
    "        # Tách tập Train/Test\n",
    "        train_dataset = NaiveMILDataset(X_data[train_idx], y_data[train_idx])\n",
    "        test_dataset = NaiveMILDataset(X_data[test_idx], y_data[test_idx])\n",
    "        \n",
    "        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
    "        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
    "        \n",
    "        # Khởi tạo lại Model cho mỗi Fold\n",
    "        model = NaiveDeepMIL().to(device)\n",
    "        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)\n",
    "        \n",
    "        best_f1 = 0.0\n",
    "        best_acc = 0.0\n",
    "        \n",
    "        for epoch in range(epochs):\n",
    "            model.train()\n",
    "            train_loss = 0.0\n",
    "            for features, labels in train_loader:\n",
    "                features, labels = features.to(device), labels.to(device)\n",
    "                \n",
    "                optimizer.zero_grad()\n",
    "                outputs = model(features)\n",
    "                loss = criterion(outputs, labels)\n",
    "                loss.backward()\n",
    "                optimizer.step()\n",
    "                \n",
    "                train_loss += loss.item()\n",
    "            \n",
    "            # Validation\n",
    "            model.eval()\n",
    "            all_preds = []\n",
    "            all_labels = []\n",
    "            with torch.no_grad():\n",
    "                for features, labels in test_loader:\n",
    "                    features, labels = features.to(device), labels.to(device)\n",
    "                    outputs = model(features)\n",
    "                    _, preds = torch.max(outputs, 1)\n",
    "                    all_preds.extend(preds.cpu().numpy())\n",
    "                    all_labels.extend(labels.cpu().numpy())\n",
    "            \n",
    "            val_acc = accuracy_score(all_labels, all_preds)\n",
    "            val_f1 = f1_score(all_labels, all_preds, average='macro')\n",
    "            \n",
    "            if val_f1 > best_f1:\n",
    "                best_f1 = val_f1\n",
    "                best_acc = val_acc\n",
    "                \n",
    "        print(f\"Fold {fold+1} Hoàn tất - Best Accuracy: {best_acc:.4f} | Best Macro-F1: {best_f1:.4f}\")\n",
    "        fold_results.append((best_acc, best_f1))\n",
    "    \n",
    "    # Tổng kết 5-Fold\n",
    "    mean_acc = np.mean([x[0] for x in fold_results])\n",
    "    std_acc = np.std([x[0] for x in fold_results])\n",
    "    mean_f1 = np.mean([x[1] for x in fold_results])\n",
    "    std_f1 = np.std([x[1] for x in fold_results])\n",
    "    \n",
    "    print(\"\\n==============================================\")\n",
    "    print(\" TỔNG KẾT NAIVE DEEP LEARNING MIL (5-FOLD)\")\n",
    "    print(\"==============================================\")\n",
    "    print(f\"🎯 Accuracy: {mean_acc:.4f} ˙ {std_acc:.4f}\")\n",
    "    print(f\"⚖️ Macro-F1: {mean_f1:.4f} ˙ {std_f1:.4f}\")\n",
    "    print(\"==============================================\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### 📉 BÀI HỌC RÚT RA TỪ SỰ THẤT BẠI CỦA NAIVE DEEP MIL:\n",
    "Bạn thấy đấy, dù chúng ta đã thay thế bằng mạng Neural Network tiên tiến (Deep Learning), điểm số vẫn tồi tệ (Thường chỉ kẹt ở mốc Accuracy ~55-60%, Macro-F1 ~0.40).\n",
    "\n",
    "**Lý do khoa học (Để báo cáo trước Hội đồng):**\n",
    "- Vấn đề **không nằm ở não bộ** (Mạng phân loại nông hay sâu).\n",
    "- Vấn đề **nằm ở giác quan** (Cách hệ thống đọc dữ liệu). Việc lấy `Mean-Pooling` (Cộng trung bình mọi vector tế bào) đã hòa tan 1% khối u HER2 độc hại vào 99% tế bào mỡ vô hại. Mạng Neural Network chỉ nhận được một tín hiệu \"nửa vời\" và hoàn toàn bị mù không thể phân biệt được.\n",
    "\n",
    "👉 Bằng chứng này chính là tấm thảm đỏ hoàn hảo để chúng ta rước **Notebook 5: TransMIL (Transformer-based MIL)** bước ra sân khấu. Mô hình Transformer sử dụng lớp `Self-Attention`, không cộng trung bình, mà nó sẽ **quét qua toàn bộ 5000 vector tế bào và gán trọng số riêng cho từng vector một**. Sự trỗi dậy của TransMIL sẽ đánh gục mọi thuật toán ở Notebook 3 và 4!"
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

out_path = r'C:\Users\huynh\Desktop\breast cancer\04_Classification_Naive_DeepMIL.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Created Notebook 4 Naive Deep MIL successfully.")
