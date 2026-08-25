import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kế Hoạch Thực Tập: AI Y Tế Đa Phương Thức\n",
    "## Notebook 2: Trích Xuất Đặc Trưng CNN (ResNet50) & Thiết Lập Baseline\n",
    "\n",
    "**Mục tiêu của Notebook này:**\n",
    "1. **Lọc dữ liệu chuẩn vàng:** Loại bỏ các ca thiếu nhãn và nhóm `BRCA_Normal` (mô nhiễu), giữ lại chính xác **945 ca bệnh** thuộc 4 phân nhóm ác tính cốt lõi.\n",
    "2. **Data Engineering (Chuyển đổi dữ liệu):** Ảnh gigapixel WSI không thể đưa trực tiếp vào mạng MIL. Chúng ta sử dụng **ResNet50** làm bộ trích xuất đặc trưng (Feature Extractor).\n",
    "3. **Tối ưu hóa Phần cứng:** Vận hành tối đa 2 GPU T4 của nền tảng Kaggle thông qua cơ chế `DataParallel`.\n",
    "4. **Xây dựng Baseline Truyền thống:** Sử dụng Mean-Pooling và Random Forest để tạo ra điểm chuẩn so sánh. Phân tích điểm yếu của phương pháp này làm bàn đạp cho kiến trúc TransMIL ở Notebook 3."
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
    "from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix\n",
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
    "# Đọc Data\n",
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
    "**Tại sao phải resize về `224x224` và chuẩn hóa (Normalize)?**\n",
    "- ResNet50 được huấn luyện trước (Pre-trained) trên tập dữ liệu ImageNet với kích thước chuẩn là `224x224` pixel.\n",
    "- Dải màu của các patch y tế (màu hồng/tím của H&E) khác với ảnh tự nhiên (chó mèo, ô tô). Tham số `mean=[0.485, 0.456, 0.406]` và `std=[0.229, 0.224, 0.225]` là phân bố màu sắc chuẩn của ImageNet. Chúng ta dùng chung bộ chuẩn hóa này để các lớp ConvNet của ResNet50 kích hoạt đúng nhất."
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
    "            image = Image.new('RGB', (224, 224), (255, 255, 255))\n",
    "            \n",
    "        if self.transform:\n",
    "            image = self.transform(image)\n",
    "        return image\n",
    "\n",
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
    "### 3. Cấu Hình Mô Hình Trích Xuất (ResNet50 Feature Extractor)\n",
    "**Lý thuyết cấu hình (Model Configuration):**\n",
    "- **Tại sao chọn ResNet50?** ResNet50 có kiến trúc Residual Connections (kết nối tắt) giúp chống lại hiện tượng suy biến đạo hàm (vanishing gradient), rất phù hợp để nhận diện các kết cấu vi thể phức tạp (hạt nhân, viền màng tế bào) vốn có độ tương phản thấp.\n",
    "- **Loại bỏ lớp cuối (Truncation):** Mạng ResNet50 nguyên bản có lớp cuối cùng dự đoán 1000 đồ vật (Fully Connected Layer). Chúng ta sẽ **cắt bỏ lớp này** (`nn.Sequential(*list(model.children())[:-1]`). Nhờ vậy, đầu ra của ảnh không phải là một nhãn đồ vật, mà là một **vector toán học 2048 chiều** đại diện cho toàn bộ thông tin sinh học của mảng tế bào đó.\n",
    "- **Thông số Dataloader:** `batch_size=256` và `num_workers=2` kết hợp `pin_memory=True` là cấu hình tối ưu nhất (Sweet-spot) để GPU T4 không bị \"đói\" dữ liệu từ CPU."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_feature_extractor():\n",
    "    model = models.resnet50(pretrained=True)\n",
    "    # Cắt lớp Linear dự đoán cuối cùng\n",
    "    model = nn.Sequential(*list(model.children())[:-1])\n",
    "    \n",
    "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "    if torch.cuda.device_count() > 1:\n",
    "        print(f\"Tối ưu hóa: Khởi động DataParallel trên {torch.cuda.device_count()} GPUs.\")\n",
    "        model = nn.DataParallel(model)\n",
    "        \n",
    "    model = model.to(device)\n",
    "    model.eval() # Bắt buộc phải set eval() để khóa Dropout và BatchNorm\n",
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
    "    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)\n",
    "    \n",
    "    features_list = []\n",
    "    with torch.no_grad(): # Tắt tính toán đạo hàm để tiết kiệm RAM\n",
    "        for images in loader:\n",
    "            images = images.to(device)\n",
    "            features = model(images)  # [Batch, 2048, 1, 1]\n",
    "            features = features.view(features.size(0), -1)  # Kéo thẳng thành [Batch, 2048]\n",
    "            features_list.append(features.cpu())\n",
    "            \n",
    "    patient_tensor = torch.cat(features_list, dim=0) # Tổng hợp lại thành [Số Patch, 2048]\n",
    "    return patient_tensor\n",
    "\n",
    "model, device = build_feature_extractor()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Vòng Lặp Xử Lý Toàn Bộ Dữ Liệu (Full Execution)\n",
    "Thiết lập `limit = None` để bắt đầu trích xuất đặc trưng cho **tất cả 945 bệnh nhân**. Mọi vector sinh ra sẽ được lưu thành file `.pt` chuẩn bị cho Notebook 3 & 4."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "valid_patients = [p for p in df_filtered['patientId'].tolist() if os.path.exists(os.path.join(WSI_DIR, str(p)))]\n",
    "print(f\"Tổng số bệnh nhân hợp lệ cần xử lý: {len(valid_patients)}\")\n",
    "\n",
    "limit = None # Chạy Toàn Bộ Data\n",
    "patients_to_process = valid_patients[:limit] if limit else valid_patients\n",
    "\n",
    "print(f\"Bắt đầu trích xuất cho {len(patients_to_process)} bệnh nhân...\")\n",
    "extracted_count = 0\n",
    "for pid in tqdm(patients_to_process):\n",
    "    out_path = os.path.join(OUTPUT_PT_DIR, f\"{pid}.pt\")\n",
    "    if not os.path.exists(out_path):\n",
    "        tensor = extract_patient_features(str(pid), model, device)\n",
    "        if tensor is not None:\n",
    "            torch.save(tensor, out_path)\n",
    "            extracted_count += 1\n",
    "\n",
    "print(f\"\\nHoàn tất! Đã trích xuất {extracted_count} file .pt mới.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Baseline Cổ Điển: Mean-Pooling & Random Forest\n",
    "\n",
    "Trước khi áp dụng siêu mạng Transformer, chuẩn mực nghiên cứu luôn yêu cầu xây dựng một mô hình cơ bản (Baseline) để đối chiếu hiệu năng.\n",
    "\n",
    "**Cách hoạt động của Baseline:**\n",
    "1. **Mean-Pooling:** Một bệnh nhân có thể có tới 5,000 patches (kích thước `[5000, 2048]`). Thuật toán Machine Learning không thể đọc ma trận 2D này. Giải pháp duy nhất của truyền thống là cộng tất cả 5,000 vector lại và chia trung bình (Mean-Pooling) để ra 1 vector duy nhất `[1, 2048]` đại diện cho cả bệnh nhân.\n",
    "   - *Yếu điểm:* Việc lấy trung bình vô tình làm \"loãng\" các đặc điểm hiếm. Nếu một ảnh WSI 10,000 patches chỉ có 10 patches ác tính, việc chia trung bình cho 9,990 patches mô mỡ sẽ làm tín hiệu ác tính bị triệt tiêu hoàn toàn.\n",
    "2. **Lựa chọn Thuật toán:** Chúng ta sử dụng **Random Forest** với `n_estimators=100` (100 cây quyết định). RF chống Overfitting rất tốt trên không gian đặc trưng lớn (2048 chiều).\n",
    "3. **Cứu vớt dữ liệu lệch (Imbalance):** Tham số `class_weight='balanced'` ép mô hình phạt nặng (penalty) gấp nhiều lần nếu đoán sai các nhóm thiểu số (như HER2), giúp mô hình không bị thiên vị mù quáng vào nhóm đông (LumA)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "label_map = {'BRCA_LumA': 0, 'BRCA_LumB': 1, 'BRCA_Basal': 2, 'BRCA_Her2': 3}\n",
    "df_filtered['label'] = df_filtered['pam50_subtype'].map(label_map)\n",
    "patient_label_dict = dict(zip(df_filtered['patientId'], df_filtered['label']))\n",
    "\n",
    "X_baseline, y_baseline, patient_ids_used = [], [], []\n",
    "\n",
    "print(\"Đang nén dữ liệu bằng Mean Pooling...\")\n",
    "for pid in patients_to_process:\n",
    "    pt_path = os.path.join(OUTPUT_PT_DIR, f\"{pid}.pt\")\n",
    "    if os.path.exists(pt_path):\n",
    "        tensor = torch.load(pt_path) # [N, 2048]\n",
    "        mean_feature = torch.mean(tensor, dim=0).numpy() # Nén về [2048]\n",
    "        X_baseline.append(mean_feature)\n",
    "        y_baseline.append(patient_label_dict[pid])\n",
    "        patient_ids_used.append(pid)\n",
    "\n",
    "X_baseline = np.array(X_baseline)\n",
    "y_baseline = np.array(y_baseline)\n",
    "\n",
    "if len(X_baseline) > 10:\n",
    "    print(f\"Hoàn tất! Kích thước Ma trận Huấn luyện Baseline: X={X_baseline.shape}, y={y_baseline.shape}\")\n",
    "    \n",
    "    # Chia 80/20. Bắt buộc dùng stratify để giữ đúng tỷ lệ 4 nhãn trong cả Train và Test\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X_baseline, y_baseline, test_size=0.2, stratify=y_baseline, random_state=42\n",
    "    )\n",
    "    \n",
    "    print(\"\\nĐang huấn luyện Random Forest Classifier...\")\n",
    "    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)\n",
    "    rf_model.fit(X_train, y_train)\n",
    "    \n",
    "    y_pred = rf_model.predict(X_test)\n",
    "    \n",
    "    print(\"\\n==============================================\")\n",
    "    print(\"      KẾT QUẢ BASELINE (TRUYỀN THỐNG)\")\n",
    "    print(\"==============================================\")\n",
    "    print(f\"🎯 Accuracy (Độ chính xác tổng): {accuracy_score(y_test, y_pred):.4f}\")\n",
    "    print(f\"⚖️ Macro-F1 (Điểm công bằng cho 4 nhãn): {f1_score(y_test, y_pred, average='macro'):.4f}\")\n",
    "    print(\"----------------------------------------------\")\n",
    "    \n",
    "    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]\n",
    "    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))\n",
    "else:\n",
    "    print(\"Lỗi: Không đủ dữ liệu file .pt để chạy Baseline.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### 🏆 TỔNG KẾT NOTEBOOK 2:\n",
    "**Bài học từ kết quả Baseline:** \n",
    "Bạn sẽ nhận thấy Macro-F1 khá thấp (đặc biệt là Recall của HER2). Nguyên nhân cốt lõi là do **Mean-Pooling** đã làm mất đi sự kết nối không gian (spatial connections) và làm \"chìm\" các mảnh tế bào mang đặc tính phân bào điển hình của khối u. \n",
    "\n",
    "👉 **Đó chính là lý do TransMIL ra đời.** Ở Notebook 3, thay vì lấy trung bình thô bạo, chúng ta sẽ để cho AI (Self-Attention của Transformer) tự động quét và đánh trọng số (Attention Scores) cho những vị trí quan trọng nhất của bệnh nhân."
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

print("Updated Notebook 2 for FULL data successfully.")
