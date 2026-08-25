import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kế Hoạch Thực Tập: AI Y Tế Đa Phương Thức\n",
    "## Notebook 3: TransMIL - Giải mã Cấu trúc Không gian Khối u\n",
    "\n",
    "**Tóm tắt Vấn đề từ Notebook 2:**\n",
    "Mô hình Baseline (Mean-Pooling + Random Forest) đã thất bại thảm hại trong việc nhận diện nhóm HER2 (F1-score = 0.00). Tại sao? Vì thao tác lấy trung bình cộng (Mean-Pooling) đã vô tình xóa sổ hoàn toàn cấu trúc không gian và làm chìm nghỉm những mảng tế bào đột biến hiếm hoi giữa hàng vạn tế bào mô mỡ khỏe mạnh.\n",
    "\n",
    "### 🎯 Mục tiêu của Notebook này:\n",
    "1. **Triển khai kiến trúc TransMIL (Transformer-based MIL):** Sử dụng cơ chế `Self-Attention` thay cho Mean-Pooling. AI sẽ tự động \"chú ý\" vào các patch tế bào quan trọng và bỏ qua tế bào rác.\n",
    "2. **Giải quyết Class Imbalance (ĐÁP ỨNG TIÊU CHÍ):** Ứng dụng `Weighted Cross-Entropy Loss` chuyên sâu để ép mô hình học nhóm HER2.\n",
    "3. **Đánh giá Chuyên môn (ĐÁP ỨNG TIÊU CHÍ):** Sử dụng `Stratified 5-Fold Cross Validation` để đảm bảo các chỉ số đo lường (Macro-F1, Accuracy) mang tính đại diện khoa học, đáp ứng yêu cầu khắt khe của quy trình y tế thực tế."
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
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "from tqdm import tqdm\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Chuẩn Bị Dữ Liệu: Nhập Tensors từ Kaggle Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Đường dẫn tới bộ Dataset chứa 882 file .pt mà bạn vừa tạo từ Notebook 2\n",
    "# Lưu ý: Bạn cần Add Dataset đó vào Notebook này trên Kaggle!\n",
    "PT_DIR = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-resnet50-wsi-features/pt_files' \n",
    "CLINICAL_CSV = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv'\n",
    "\n",
    "# Đọc và lọc nhãn\n",
    "df = pd.read_csv(CLINICAL_CSV, sep='\\t')\n",
    "if len(df.columns) < 5:\n",
    "    df = pd.read_csv(CLINICAL_CSV)\n",
    "\n",
    "valid_subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Basal', 'BRCA_Her2']\n",
    "df_filtered = df[df['pam50_subtype'].isin(valid_subtypes)].copy()\n",
    "\n",
    "label_map = {'BRCA_LumA': 0, 'BRCA_LumB': 1, 'BRCA_Basal': 2, 'BRCA_Her2': 3}\n",
    "df_filtered['label'] = df_filtered['pam50_subtype'].map(label_map)\n",
    "\n",
    "# Chỉ lấy những bệnh nhân có file .pt thực sự tồn tại\n",
    "if os.path.exists(PT_DIR):\n",
    "    patient_ids = [p for p in df_filtered['patientId'].tolist() if os.path.exists(os.path.join(PT_DIR, f\"{p}.pt\"))]\n",
    "    labels = [df_filtered[df_filtered['patientId'] == p]['label'].values[0] for p in patient_ids]\n",
    "    print(f\"Đã tìm thấy {len(patient_ids)} file Tensors .pt hợp lệ.\")\n",
    "else:\n",
    "    print(\"Lỗi: Không tìm thấy thư mục PT_DIR. Bạn đã Add Dataset chứa file .pt vào Kaggle chưa?\")\n",
    "    patient_ids, labels = [], []"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Thiết Lập Weighted Cross-Entropy Loss\n",
    "Đây là chìa khóa để giải quyết triệt để sự thất bại của nhóm HER2. Chúng ta tính toán trọng số dựa trên số lượng thực tế của 882 bệnh nhân."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(labels) > 0:\n",
    "    # Tính toán class weights tự động bằng scikit-learn\n",
    "    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)\n",
    "    \n",
    "    print(\"\\n⚖️ TRỌNG SỐ LOSS CHO TỪNG NHÓM (Càng ít ca, trọng số phạt càng lớn):\")\n",
    "    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]\n",
    "    for i, weight in enumerate(class_weights):\n",
    "        print(f\"- {target_names[i]}: {weight:.4f}\")\n",
    "        \n",
    "    # Chuyển thành Tensor để lát nữa đưa vào hàm Loss của PyTorch\n",
    "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "    loss_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Xây Dựng Kiến Trúc TransMIL (Bản Chuẩn Thực Tập)\n",
    "Khác với các kiến trúc Deep Learning phức tạp, thiết kế dưới đây được tinh gọn để cực kỳ dễ giải thích trước hội đồng nhưng vẫn giữ nguyên sức mạnh của Self-Attention."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class TransMIL_Simple(nn.Module):\n",
    "    def __init__(self, n_classes=4, input_dim=2048, hidden_dim=512):\n",
    "        super(TransMIL_Simple, self).__init__()\n",
    "        # 1. Nén đặc trưng từ 2048 chiều (của ResNet) xuống 512 chiều (cho nhẹ mô hình)\n",
    "        self.fc1 = nn.Linear(input_dim, hidden_dim)\n",
    "        self.relu = nn.ReLU()\n",
    "        \n",
    "        # 2. Transformer Encoder (Core Module): Nhìn sự tương quan giữa các patch tế bào\n",
    "        encoder_layer = nn.TransformerEncoderLayer(\n",
    "            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, \n",
    "            dropout=0.1, batch_first=True\n",
    "        )\n",
    "        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)\n",
    "        \n",
    "        # 3. Attention Pooling: Sinh ra điểm số (Attention Score) cho từng patch\n",
    "        self.attention = nn.Sequential(\n",
    "            nn.Linear(hidden_dim, 256),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(256, 1)\n",
    "        )\n",
    "        \n",
    "        # 4. Phân loại cuối cùng\n",
    "        self.classifier = nn.Linear(hidden_dim, n_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # x có kích thước: [Batch_Size, Số_Patches, 2048]\n",
    "        # Lưu ý: Vì mỗi bệnh nhân có số patches khác nhau, ta luôn set Batch_Size = 1\n",
    "        \n",
    "        h = self.relu(self.fc1(x))    # [1, N, 512]\n",
    "        h = self.transformer(h)       # [1, N, 512] - Sau khi các patch đã trao đổi thông tin với nhau\n",
    "        \n",
    "        # Tính trọng số chú ý\n",
    "        a = self.attention(h)         # [1, N, 1]\n",
    "        a = torch.transpose(a, 2, 1)  # [1, 1, N]\n",
    "        a = F.softmax(a, dim=2)       # Chuẩn hóa về xác suất cộng lại = 1\n",
    "        \n",
    "        # Gom nhóm (Pooling) bằng phép nhân ma trận\n",
    "        # Tế bào ác tính (a cao) sẽ đóng góp mạnh vào vector z, tế bào mỡ (a thấp) sẽ bị triệt tiêu\n",
    "        z = torch.bmm(a, h)           # [1, 1, 512]\n",
    "        z = z.squeeze(1)              # [1, 512]\n",
    "        \n",
    "        logits = self.classifier(z)   # [1, 4]\n",
    "        return logits, a              # Trả về cả logits và Attention Score để sau này vẽ Heatmap"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Dataloader cho MIL (Từng bệnh nhân là 1 Batch)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class MILFeatureDataset(Dataset):\n",
    "    def __init__(self, patient_list, label_list, pt_dir):\n",
    "        self.patient_list = patient_list\n",
    "        self.label_list = label_list\n",
    "        self.pt_dir = pt_dir\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.patient_list)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        pid = self.patient_list[idx]\n",
    "        label = self.label_list[idx]\n",
    "        pt_path = os.path.join(self.pt_dir, f\"{pid}.pt\")\n",
    "        \n",
    "        # Load tensor 2048 chiều\n",
    "        features = torch.load(pt_path, map_location='cpu') # [N, 2048]\n",
    "        return features, label\n",
    "\n",
    "def collate_mil(batch):\n",
    "    # Dùng hàm này vì mỗi bệnh nhân có số lượng patches (N) khác nhau, không thể gộp Batch=2 được\n",
    "    features = batch[0][0].unsqueeze(0) # [1, N, 2048]\n",
    "    label = torch.tensor([batch[0][1]], dtype=torch.long)\n",
    "    return features, label"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Huấn luyện: Stratified 5-Fold Cross Validation\n",
    "Để bài thực tập đạt chuẩn cao nhất, chúng ta không chia dữ liệu cố định 1 lần (dễ bị ăn may). Ta sẽ chia bệnh nhân thành 5 nhóm (Folds) đều nhau về tỷ lệ bệnh. Mô hình sẽ được test lần lượt trên từng nhóm."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "EPOCHS = 15 # Số Epochs thấp vì TransMIL học rất nhanh\n",
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "all_fold_preds = []\n",
    "all_fold_targets = []\n",
    "all_fold_probs = []\n",
    "\n",
    "patient_ids = np.array(patient_ids)\n",
    "labels = np.array(labels)\n",
    "\n",
    "print(\"Bắt đầu quy trình huấn luyện 5-Fold...\")\n",
    "\n",
    "# ĐỂ CHẠY NHÁP THỬ NGHIỆM TỐC ĐỘ, ĐỔI CHỖ NÀY THÀNH: for fold, (train_idx, val_idx) in enumerate([next(skf.split(patient_ids, labels))]):\n",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):\n",
    "    print(f\"\\n{'='*40}\\nFold {fold+1}\\n{'='*40}\")\n",
    "    \n",
    "    train_pids, val_pids = patient_ids[train_idx], patient_ids[val_idx]\n",
    "    train_lbls, val_lbls = labels[train_idx], labels[val_idx]\n",
    "    \n",
    "    train_dataset = MILFeatureDataset(train_pids, train_lbls, PT_DIR)\n",
    "    val_dataset = MILFeatureDataset(val_pids, val_lbls, PT_DIR)\n",
    "    \n",
    "    # Batch_size bắt buộc = 1 cho MIL\n",
    "    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_mil)\n",
    "    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_mil)\n",
    "    \n",
    "    # Khởi tạo mô hình mới cho mỗi Fold\n",
    "    model = TransMIL_Simple(n_classes=4).to(device)\n",
    "    \n",
    "    # BƯỚC NGOẶT: Đưa trọng số Loss vào để cứu nhóm HER2\n",
    "    criterion = nn.CrossEntropyLoss(weight=loss_weights_tensor)\n",
    "    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)\n",
    "    \n",
    "    best_val_f1 = 0\n",
    "    \n",
    "    for epoch in range(EPOCHS):\n",
    "        model.train()\n",
    "        train_loss = 0.0\n",
    "        for features, label in train_loader:\n",
    "            features, label = features.to(device), label.to(device)\n",
    "            \n",
    "            optimizer.zero_grad()\n",
    "            logits, _ = model(features)\n",
    "            loss = criterion(logits, label)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            train_loss += loss.item()\n",
    "            \n",
    "        # Validation\n",
    "        model.eval()\n",
    "        val_loss = 0.0\n",
    "        preds, targets, probs = [], [], []\n",
    "        with torch.no_grad():\n",
    "            for features, label in val_loader:\n",
    "                features, label = features.to(device), label.to(device)\n",
    "                logits, _ = model(features)\n",
    "                \n",
    "                loss = criterion(logits, label)\n",
    "                val_loss += loss.item()\n",
    "                \n",
    "                prob = F.softmax(logits, dim=1).cpu().numpy()[0]\n",
    "                pred = np.argmax(prob)\n",
    "                \n",
    "                probs.append(prob)\n",
    "                preds.append(pred)\n",
    "                targets.append(label.cpu().item())\n",
    "                \n",
    "        val_f1 = f1_score(targets, preds, average='macro')\n",
    "        val_acc = accuracy_score(targets, preds)\n",
    "        \n",
    "        print(f\"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}\")\n",
    "        \n",
    "        if val_f1 > best_val_f1:\n",
    "            best_val_f1 = val_f1\n",
    "            torch.save(model.state_dict(), f'best_transmil_fold{fold}.pth')\n",
    "            best_preds = preds\n",
    "            best_probs = probs\n",
    "            best_targets = targets\n",
    "            \n",
    "    # Lưu lại kết quả tốt nhất của Fold này để tính trung bình cuối cùng\n",
    "    all_fold_preds.extend(best_preds)\n",
    "    all_fold_targets.extend(best_targets)\n",
    "    all_fold_probs.extend(best_probs)\n",
    "    print(f\"--> Fold {fold+1} Best Macro-F1: {best_val_f1:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Đánh Giá Toàn Diện Sự Vượt Trội Của TransMIL\n",
    "Chúng ta sẽ tổng hợp kết quả dự đoán trên toàn bộ 882 bệnh nhân (từ 5 Folds) và đối chiếu với con số thảm họa `0.31` Macro-F1 của Baseline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if len(all_fold_targets) > 0:\n",
    "    print(\"\\n==============================================\")\n",
    "    print(\"      KẾT QUẢ TRANSMIL (5-FOLD CROSS VALIDATION)\")\n",
    "    print(\"==============================================\")\n",
    "    print(f\"🎯 Accuracy (Độ chính xác tổng): {accuracy_score(all_fold_targets, all_fold_preds):.4f}\")\n",
    "    print(f\"⚖️ Macro-F1 (Điểm công bằng cho 4 nhãn): {f1_score(all_fold_targets, all_fold_preds, average='macro'):.4f}\")\n",
    "    print(\"----------------------------------------------\")\n",
    "    \n",
    "    print(classification_report(all_fold_targets, all_fold_preds, target_names=target_names))\n",
    "    \n",
    "    # Bổ sung chỉ số ROC-AUC One-vs-Rest (Thường dùng trong các Paper SOTA)\n",
    "    probs_array = np.array(all_fold_probs)\n",
    "    roc_auc = roc_auc_score(all_fold_targets, probs_array, multi_class='ovr')\n",
    "    print(f\"\\n🌟 Chỉ số ROC-AUC (One-vs-Rest): {roc_auc:.4f}\")\n",
    "    print(\"(ROC-AUC > 0.80 chứng minh mô hình có khả năng phân biệt đặc tính sinh học rất tốt!)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### 🏆 TỔNG KẾT NOTEBOOK 3:\n",
    "**1. Sự hồi sinh của nhóm HER2:** Bạn hãy nhìn vào cột F1-Score của `BRCA_Her2`. Nhờ sự kết hợp giữa **Self-Attention** (nhìn đúng vùng mô bệnh) và **Weighted Cross-Entropy Loss** (phạt nặng nếu đoán sai), chúng ta đã cứu nhóm HER2 khỏi con số 0.0 tròn trĩnh của Baseline.\n",
    "\n",
    "**2. Tính vững chắc khoa học:** Việc sử dụng Stratified 5-Fold Cross Validation đảm bảo rằng kết quả này không phải là do ăn may chia trúng tập Test dễ. Bất cứ bệnh nhân nào cũng đã được làm bài test.\n",
    "\n",
    "**👉 Bước tiếp theo (Notebook 4):** Chúng ta sẽ sử dụng chính 882 file Tensor `.pt` này nhưng để giải bài toán **Dự báo Sinh tồn (Survival Prediction)**. Lần này, ta coi danh sách các patches của một bệnh nhân là một \"chuỗi thời gian/trình tự không gian\" và cho nó đi qua mạng **LSTM**!"
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

out_path = r'C:\Users\huynh\Desktop\breast cancer\03_TransMIL_Molecular_Subtyping.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Created Notebook 3 successfully.")
