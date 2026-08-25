import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 6.5: Cấu trúc Đa phương thức sử dụng LSTM (WSI BiLSTM + Genomics MLP)\n",
    "---\n",
    "**Mục đích:** Hướng tiếp cận này sử dụng mạng **BiLSTM** (Bidirectional Long Short-Term Memory) để xử lý ảnh WSI thay vì dùng Transformer như các Notebook trước. \n",
    "\n",
    "**Cách giải thích bảo vệ trước Hội đồng:**\n",
    "> \"Thưa cô, khối u của bệnh nhân quá lớn nên phải cắt thành hàng ngàn mảnh nhỏ (patches). Em coi chuỗi hàng ngàn mảnh ảnh này giống như một 'chuỗi không gian' (Spatial Sequence). Thay vì dùng CNN thông thường không thể xử lý ảnh to, em dùng mạng BiLSTM. LSTM sẽ trượt qua toàn bộ các mảnh ảnh, ghi nhớ (Memory) và tích lũy các đặc trưng tế bào ung thư từ mảnh ảnh này sang mảnh ảnh khác. Sau khi trượt xong, em tính trung bình (Global Average) các kết quả của LSTM để tạo ra 1 vector duy nhất đại diện cho toàn bộ khối u của bệnh nhân. Sau đó em mới nối (Fusion) nó với dữ liệu Gen (đã qua MLP) để đưa ra dự đoán cuối cùng.\"\n",
    "\n",
    "Cách làm này cực kỳ **đơn giản, code ngắn gọn, dễ hiểu** mà vẫn hoàn toàn đáp ứng đúng yêu cầu của bộ môn về việc ứng dụng LSTM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Fix font tiếng Việt\n",
    "plt.rcParams['font.family'] = 'sans-serif'\n",
    "plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans']\n",
    "plt.rcParams['axes.unicode_minus'] = False\n",
    "\n",
    "print(\"✓ Khởi tạo môi trường thành công.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Định nghĩa Mạng BiLSTM siêu tối giản cho WSI\n",
    "Mạng này chỉ gồm 3 bước:\n",
    "1. Ép chiều dữ liệu từ 2048 xuống 512.\n",
    "2. Đưa qua hàm `nn.LSTM` với tham số `bidirectional=True` (quét từ 2 phía để không bỏ sót thông tin).\n",
    "3. Gộp tất cả các mảnh ảnh lại bằng phép toán Trung bình (Mean) siêu đơn giản."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Simple_BiLSTM_Vision(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        # 1. Ép chiều dữ liệu ảnh gốc từ 2048 chiều xuống 512 chiều\n",
    "        self.fc = nn.Linear(2048, 512)\n",
    "        \n",
    "        # 2. Khai báo LSTM: 512 đầu vào, 256 đầu ra. Vì bidirectional=True nên đầu ra thực tế là 256*2 = 512.\n",
    "        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, \n",
    "                            batch_first=True, bidirectional=True)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        # Input x: [Batch_size, Số_mảnh_ảnh, 2048]\n",
    "        \n",
    "        # Bước 1: Ép chiều\n",
    "        x = self.fc(x) # -> [Batch_size, Số_mảnh_ảnh, 512]\n",
    "        \n",
    "        # Bước 2: Đi qua LSTM\n",
    "        # lstm_out chứa kết quả của từng mảnh ảnh sau khi đã ghi nhớ thông tin xung quanh\n",
    "        lstm_out, (hn, cn) = self.lstm(x) # lstm_out: [Batch_size, Số_mảnh_ảnh, 512]\n",
    "        \n",
    "        # Bước 3: Gộp thông tin bằng Trung bình (Global Average Pooling)\n",
    "        # Tính trung bình trên chiều số 1 (chiều của các mảnh ảnh)\n",
    "        final_wsi_vector = lstm_out.mean(dim=1) # -> [Batch_size, 512]\n",
    "        \n",
    "        return final_wsi_vector\n",
    "\n",
    "print(\"✓ Khởi tạo kiến trúc Simple_BiLSTM_Vision thành công.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Nhánh Xử lý Gen (Tabular MLP) và Cầu nối Late Fusion\n",
    "Phần này giữ nguyên như kiến trúc xuất sắc của Notebook 6.2, đảm bảo dữ liệu Gen vẫn được ép về không gian 512 chiều bằng mạng MLP đơn giản."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Simple_Genomics_MLP(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(500, 256),\n",
    "            nn.LayerNorm(256),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(0.3),\n",
    "            nn.Linear(256, 512),\n",
    "            nn.ReLU()\n",
    "        )\n",
    "    def forward(self, x):\n",
    "        return self.net(x)  # -> [Batch_size, 512]\n",
    "\n",
    "class Multimodal_BiLSTM_Fusion(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.vision_lstm = Simple_BiLSTM_Vision()\n",
    "        self.genomics_mlp = Simple_Genomics_MLP()\n",
    "        \n",
    "        # Nhận vào 512 (Vision) + 512 (Gen) = 1024 chiều\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(1024, 256),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(256, 4) # 4 phân nhóm ung thư PAM50\n",
    "        )\n",
    "        \n",
    "    def forward(self, wsi, rna):\n",
    "        v_feat = self.vision_lstm(wsi)  # Trích xuất 512 chiều ảnh qua LSTM\n",
    "        g_feat = self.genomics_mlp(rna) # Trích xuất 512 chiều gen qua MLP\n",
    "        \n",
    "        # LATE FUSION: Nối 2 vector lại với nhau\n",
    "        fusion = torch.cat((v_feat, g_feat), dim=1) # -> 1024 chiều\n",
    "        \n",
    "        # Phân loại cuối cùng\n",
    "        out = self.classifier(fusion) # -> 4 chiều\n",
    "        return out\n",
    "\n",
    "print(\"✓ Khởi tạo kiến trúc Fusion hoàn chỉnh thành công.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Chạy thử nghiệm Dummy (Kiểm tra hình dáng Tensor)\n",
    "Đoạn code này mô phỏng đưa dữ liệu của 2 bệnh nhân (Batch=2) vào mô hình để chứng minh kiến trúc hoạt động hoàn hảo, không bị lỗi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Khởi tạo mô hình\n",
    "model = Multimodal_BiLSTM_Fusion()\n",
    "\n",
    "# Giả lập dữ liệu cho 2 bệnh nhân\n",
    "# Bệnh nhân 1 có 800 mảnh ảnh, Bệnh nhân 2 có 800 mảnh ảnh (đã được pad_collate)\n",
    "dummy_wsi = torch.randn(2, 800, 2048) \n",
    "dummy_rna = torch.randn(2, 500)       # 500 gen\n",
    "\n",
    "print(\"📥 Tensor Đầu vào:\")\n",
    "print(f\" - WSI Tensor: {dummy_wsi.shape}\")\n",
    "print(f\" - Gen Tensor: {dummy_rna.shape}\\n\")\n",
    "\n",
    "# Chạy qua mô hình\n",
    "output = model(dummy_wsi, dummy_rna)\n",
    "\n",
    "print(\"📤 Tensor Đầu ra (Logits dự đoán cho 4 class PAM50):\")\n",
    "print(output)\n",
    "print(f\"=> Kích thước: {output.shape} (Quá chuẩn!)\")\n",
    "\n",
    "print(\"\\n🎉 XONG! CHỈ VỚI ĐOẠN CODE NGẮN GỌN NÀY, BẠN ĐÃ ĐÁP ỨNG ĐƯỢC YÊU CẦU DÙNG LSTM CỦA GIẢNG VIÊN.\")"
   ]
  }
 ]
}

with open(r'C:\Users\huynh\Desktop\breast cancer\06.5_Multimodal_BiLSTM_Genomics_Simple.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print("Đã tạo file 06.5_Multimodal_BiLSTM_Genomics_Simple.ipynb")
