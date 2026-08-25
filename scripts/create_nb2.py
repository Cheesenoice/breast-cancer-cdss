import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kế Hoạch Thực Tập: AI Y Tế Đa Phương Thức\n",
    "## Notebook 2: Trích xuất Đặc trưng CNN (ResNet50) & Xây dựng Baseline\n",
    "\n",
    "**Mục tiêu của Notebook này:**\n",
    "1. **Lọc dữ liệu chuẩn vàng:** Dựa trên phân tích từ Notebook 1, chúng ta sẽ loại bỏ các ca bệnh không có nhãn hoặc thuộc nhóm `BRCA_Normal`. Giữ lại chính xác **945 ca bệnh** thuộc 4 phân nhóm: LumA, LumB, Basal, Her2.\n",
    "2. **Chuyển đổi Dữ liệu (Data Engineering):** Ảnh WSI quá lớn để đưa vào mạng MIL nguyên bản. Chúng ta sẽ dùng mô hình **ResNet50** (đã được pre-train trên ImageNet) đóng vai trò như một \"con mắt\" chuyên gia để nhìn vào từng patch ảnh, và nén chúng thành một vector đặc trưng toán học (Feature Vector).\n",
    "3. **Tối ưu hóa Kaggle (DataParallel):** Tận dụng tối đa 2 card GPU T4 của Kaggle để tăng tốc độ xử lý.\n",
    "4. **Xây dựng Baseline Truyền thống:** Sử dụng phương pháp Mean-Pooling (Lấy trung bình cộng các patch) kết hợp với thuật toán Machine Learning cổ điển (Random Forest) để thiết lập một điểm chuẩn (Baseline) trước khi dùng đến siêu kiến trúc TransMIL."
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
    "from PIL import Image\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from torchvision import transforms, models\n",
    "from tqdm import tqdm\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, f1_score, accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Nạp và Lọc Dữ Liệu Lâm Sàng (Chuẩn Bị Nhãn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Đường dẫn Kaggle\n",
    "DATA_DIR = '/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis'\n",
    "WSI_DIR = os.path.join(DATA_DIR, 'WSIs')\n",
    "CLINICAL_CSV = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca/tcga_brca_master_matched_cohort.csv'\n",
    "OUTPUT_PT_DIR = '/kaggle/working/pt_files'  # Nơi lưu Tensor\n",
    "os.makedirs(OUTPUT_PT_DIR, exist_ok=True)\n",
    "\n",
    "# Đọc và làm sạch Data\n",
    "df = pd.read_csv(CLINICAL_CSV, sep='\\t')\n",
    "if len(df.columns) < 5:\n",
    "    df = pd.read_csv(CLINICAL_CSV)\n",
    "\n",
    "# LOẠI BỎ Missing và Nhóm Normal\n",
    "valid_subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Basal', 'BRCA_Her2']\n",
    "df_filtered = df[df['pam50_subtype'].isin(valid_subtypes)].copy()\n",
    "print(f\"Số lượng bệnh nhân sau khi lọc chuẩn: {len(df_filtered)} (Kỳ vọng ~945)\")\n",
    "print(df_filtered['pam50_subtype'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Thiết lập PyTorch Dataset cho WSI Patches\n",
    "Mỗi bệnh nhân (Patient) là một tập hợp (Bag) của hàng ngàn ảnh patches."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class PatientPatchDataset(Dataset):\n",
    "    def __init__(self, patient_dir, transform=None):\n",
    "        self.patient_dir = patient_dir\n",
    "        # Lấy toàn bộ file jpg\n",
    "        self.image_paths = [os.path.join(patient_dir, img) for img in os.listdir(patient_dir) if img.endswith('.jpg')]\n",
    "        self.transform = transform\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.image_paths)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        img_path = self.image_paths[idx]\n",
    "        try:\n",
    "            image = Image.open(img_path).convert('RGB')\n",
    "        except:\n",
    "            # Fallback nếu ảnh hỏng\n",
    "            image = Image.new('RGB', (224, 224), (255, 255, 255))\n",
    "            \n",
    "        if self.transform:\n",
    "            image = self.transform(image)\n",
    "        return image\n",
    "\n",
    "# ResNet50 yêu cầu normalize theo chuẩn ImageNet\n",
    "transform = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Quy trình Trích xuất Đặc trưng (Feature Extraction Pipeline)\n",
    "Sử dụng ResNet50. Để chạy siêu tốc trên Kaggle T4x2, chúng ta dùng `DataParallel` và đẩy Batch Size lên cao."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_feature_extractor():\n",
    "    # Load pre-trained ResNet50\n",
    "    model = models.resnet50(pretrained=True)\n",
    "    # Loại bỏ lớp phân loại cuối cùng (FC layer) để lấy vector đặc trưng (2048 chiều)\n",
    "    model = nn.Sequential(*list(model.children())[:-1])\n",
    "    \n",
    "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "    if torch.cuda.device_count() > 1:\n",
    "        print(f\"Tuyệt vời! Kaggle cấp cho bạn {torch.cuda.device_count()} GPUs. Đang bật DataParallel...\")\n",
    "        model = nn.DataParallel(model)\n",
    "        \n",
    "    model = model.to(device)\n",
    "    model.eval() # Bật chế độ suy luận (không huấn luyện)\n",
    "    return model, device\n",
    "\n",
    "def extract_patient_features(patient_id, model, device, batch_size=256):\n",
    "    patient_dir = os.path.join(WSI_DIR, patient_id)\n",
    "    if not os.path.exists(patient_dir):\n",
    "        return None\n",
    "        \n",
    "    dataset = PatientPatchDataset(patient_dir, transform)\n",
    "    if len(dataset) == 0:\n",
    "        return None\n",
    "        \n",
    "    # Bật num_workers để CPU đọc ảnh song song với GPU\n",
    "    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)\n",
    "    \n",
    "    features_list = []\n",
    "    with torch.no_grad():\n",
    "        for images in loader:\n",
    "            images = images.to(device)\n",
    "            features = model(images)  # Output: [Batch, 2048, 1, 1]\n",
    "            features = features.view(features.size(0), -1)  # Flatten: [Batch, 2048]\n",
    "            features_list.append(features.cpu())\n",
    "            \n",
    "    patient_tensor = torch.cat(features_list, dim=0) # [Tổng số Patches, 2048]\n",
    "    return patient_tensor\n",
    "\n",
    "model, device = build_feature_extractor()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Vòng lặp Xử lý Toàn bộ Dữ liệu\n",
    "Ở Notebook này, vì mục đích demo PoC (Proof of Concept), code sẽ chỉ chạy thử trên **20 bệnh nhân đầu tiên**. Khi bạn sẵn sàng, hãy đổi `limit = None` để chạy toàn bộ 945 bệnh nhân. Lưu ý quá trình chạy toàn bộ có thể mất vài giờ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Chỉ lấy những bệnh nhân có thư mục ảnh thực tế\n",
    "valid_patients = [p for p in df_filtered['patientId'].tolist() if os.path.exists(os.path.join(WSI_DIR, str(p)))]\n",
    "print(f\"Tổng số bệnh nhân có đủ Dữ liệu Lâm sàng & Ảnh: {len(valid_patients)}\")\n",
    "\n",
    "# ĐỂ CHẠY TOÀN BỘ, HÃY XÓA DÒNG NÀY HOẶC ĐỔI limit = None\n",
    "limit = 20 \n",
    "patients_to_process = valid_patients[:limit] if limit else valid_patients\n",
    "\n",
    "print(f\"Bắt đầu trích xuất cho {len(patients_to_process)} bệnh nhân...\")\n",
    "\n",
    "extracted_count = 0\n",
    "for pid in tqdm(patients_to_process):\n",
    "    out_path = os.path.join(OUTPUT_PT_DIR, f\"{pid}.pt\")\n",
    "    # Nếu đã chạy trước đó rồi thì bỏ qua để tiết kiệm thời gian\n",
    "    if not os.path.exists(out_path):\n",
    "        tensor = extract_patient_features(str(pid), model, device)\n",
    "        if tensor is not None:\n",
    "            torch.save(tensor, out_path)\n",
    "            extracted_count += 1\n",
    "\n",
    "print(f\"\\nĐã trích xuất và lưu {extracted_count} file .pt thành công.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Xây Dựng Baseline Cổ Điển (Random Forest + Mean Pooling)\n",
    "Trái ngược với TransMIL học từng patch độc lập, Machine Learning truyền thống bắt buộc phải nén toàn bộ ảnh WSI thành 1 vector duy nhất bằng cách tính **Giá trị Trung bình (Mean Pooling)** của tất cả các patches."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tạo một DataFrame nhỏ chứa nhãn để map\n",
    "label_map = {'BRCA_LumA': 0, 'BRCA_LumB': 1, 'BRCA_Basal': 2, 'BRCA_Her2': 3}\n",
    "df_filtered['label'] = df_filtered['pam50_subtype'].map(label_map)\n",
    "patient_label_dict = dict(zip(df_filtered['patientId'], df_filtered['label']))\n",
    "\n",
    "X_baseline = []\n",
    "y_baseline = []\n",
    "patient_ids_used = []\n",
    "\n",
    "print(\"Đang tải Tensors để lấy trung bình (Mean Pooling)...\")\n",
    "for pid in patients_to_process:\n",
    "    pt_path = os.path.join(OUTPUT_PT_DIR, f\"{pid}.pt\")\n",
    "    if os.path.exists(pt_path):\n",
    "        tensor = torch.load(pt_path) # Kích thước: [N, 2048]\n",
    "        mean_feature = torch.mean(tensor, dim=0).numpy() # Ép thành vector [2048]\n",
    "        X_baseline.append(mean_feature)\n",
    "        y_baseline.append(patient_label_dict[pid])\n",
    "        patient_ids_used.append(pid)\n",
    "\n",
    "X_baseline = np.array(X_baseline)\n",
    "y_baseline = np.array(y_baseline)\n",
    "\n",
    "if len(X_baseline) > 5:\n",
    "    print(f\"Kích thước Ma trận Baseline X: {X_baseline.shape}, y: {y_baseline.shape}\")\n",
    "    \n",
    "    # Chia Train/Test đơn giản (Cho Baseline)\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=0.2, stratify=y_baseline, random_state=42)\n",
    "    \n",
    "    print(\"Huấn luyện Random Forest...\")\n",
    "    # Chúng ta sử dụng class_weight='balanced' y như cách các paper xử lý\n",
    "    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)\n",
    "    rf_model.fit(X_train, y_train)\n",
    "    \n",
    "    y_pred = rf_model.predict(X_test)\n",
    "    print(\"\\n--- KẾT QUẢ BASELINE (TRUYỀN THỐNG) ---\")\n",
    "    print(f\"Accuracy: {accuracy_score(y_test, y_pred):.4f}\")\n",
    "    print(f\"Macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}\")\n",
    "    \n",
    "    # In Report chi tiết theo từng nhãn\n",
    "    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]\n",
    "    # Chỉ lấy các nhãn thực sự xuất hiện trong tập y_test (do đang chạy nháp 20 ca)\n",
    "    labels_present = np.unique(y_test)\n",
    "    present_target_names = [target_names[i] for i in labels_present]\n",
    "    print(\"\\nClassification Report:\")\n",
    "    print(classification_report(y_test, y_pred, target_names=present_target_names, zero_division=0))\n",
    "else:\n",
    "    print(\"Dữ liệu mẫu quá ít để huấn luyện Baseline. Hãy đổi limit lớn hơn.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### 🏆 TỔNG KẾT NOTEBOOK 2:\n",
    "1. Chúng ta đã thiết lập thành công hệ thống Dataloader để đọc ảnh WSI gigapixel bằng cách đẩy GPU vào chế độ `DataParallel`.\n",
    "2. Nén toàn bộ ảnh thành các file `.pt` toán học. Bước này giải phóng bộ nhớ RAM và giúp các mô hình AI sau này chạy mượt mà trên ma trận số.\n",
    "3. Điểm **Baseline Truyền thống (Mean Pooling + Random Forest)** có thể sẽ không cao (đặc biệt là Macro-F1 của nhóm Her2 thường sập hoàn toàn). \n",
    "4. Ở Notebook 3 (**TransMIL**), chúng ta sẽ không dùng Mean Pooling nữa mà dùng Attention của mạng Transformer kết hợp với `Weighted Cross-Entropy Loss` cực mạnh để \"giải cứu\" nhóm Her2!"
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

out_path = r'C:\Users\huynh\Desktop\breast cancer\02_CNN_Feature_Extraction_and_Baseline.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Created Notebook 2 successfully.")
