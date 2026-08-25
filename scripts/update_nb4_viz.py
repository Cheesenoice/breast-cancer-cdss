import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\04_Classification_Naive_DeepMIL.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Thêm biến lưu id bệnh nhân vào vòng lặp training
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Fold 1/5' in ''.join(cell.get('source', [])):
        # Chúng ta cần thêm biến lưu lại Patient ID để lát nữa phân tích lỗi
        old_source = cell['source']
        new_source = []
        for line in old_source:
            if "all_preds = []" in line:
                new_source.append("            all_preds = []\n")
                new_source.append("            all_labels = []\n")
                new_source.append("            all_patient_ids = [] # Thêm biến này\n")
                continue
            if "all_labels = []" in line:
                continue # Bỏ qua vì đã thêm ở trên
            if "with torch.no_grad():" in line:
                new_source.append(line)
                continue
            if "outputs = model(features)" in line:
                # Tìm cách lấy patient ids trong test_loader. 
                # Chú ý: DataLoader hiện tại không trả về ID. 
                # Để nhanh gọn, ta sẽ lưu dự đoán cuối cùng (bên ngoài vòng lặp fold) 
                # bằng cách map test_idx về patient_ids từ dữ liệu gốc X_data.
                new_source.append(line)
                continue
            new_source.append(line)
            
        # Do thay đổi DataLoader để trả về ID hơi phức tạp, ta sẽ thêm cell mới hoàn toàn 
        # chạy Inference trên toàn bộ tập dữ liệu (sau khi train xong 5-Fold) để lấy Confusion Matrix tổng và mổ xẻ lỗi.
        break

# Thêm các cell mới vào cuối notebook
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Trực Quan Hóa Ma Trận Nhầm Lẫn (Confusion Matrix)\n",
            "Để xem chi tiết mô hình đã \"đoán mò\" như thế nào, chúng ta sẽ vẽ Confusion Matrix."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "if len(X_data) > 0:\n",
            "    # Dùng mô hình ở Fold cuối cùng (hoặc train 1 mô hình tổng) để sinh dự đoán trên toàn bộ tập X_data\n",
            "    model.eval()\n",
            "    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)\n",
            "    with torch.no_grad():\n",
            "        outputs = model(X_tensor)\n",
            "        _, global_preds = torch.max(outputs, 1)\n",
            "        global_preds = global_preds.cpu().numpy()\n",
            "        \n",
            "    cm = confusion_matrix(y_data, global_preds)\n",
            "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['LumA', 'LumB', 'Basal', 'HER2'])\n",
            "    \n",
            "    fig, ax = plt.subplots(figsize=(8, 6))\n",
            "    disp.plot(cmap='Blues', ax=ax, values_format='d')\n",
            "    plt.title('Confusion Matrix - Naive Deep MIL (Mean Pooling)', fontsize=14)\n",
            "    plt.show()\n",
            "    \n",
            "    # Tìm các ca đoán sai (Ví dụ: Thực tế là HER2 nhưng đoán là LumA)\n",
            "    misclassified_idx = np.where((y_data == 3) & (global_preds != 3))[0]\n",
            "    print(f\"\\nSố lượng ca HER2 bị nhận diện sai: {len(misclassified_idx)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. Khám Nghiệm Hiện Trường (Tại sao Mean-Pooling lại giết chết Model?)\n",
            "Chúng ta sẽ lấy ngẫu nhiên 1 bệnh nhân HER2 bị mô hình nhận diện sai ở trên. Sau đó, mở thư mục chứa ảnh gốc (`.jpg`) của bệnh nhân này ra và bốc ngẫu nhiên 16 mảnh tế bào để xem chúng trông như thế nào.\n",
            "\n",
            "**👉 Giải thích phản biện:** Bạn sẽ thấy trong 16 ảnh, có thể chỉ có 1-2 ảnh chứa tế bào ung thư thực sự, còn lại toàn là mô mỡ hoặc nền trắng vô hại. Việc chúng ta dùng hàm `Mean()` (Cộng trung bình) ở bước 1 đã khiến tín hiệu của 1 tấm ảnh ung thư bị hòa tan hoàn toàn vào 15 tấm ảnh mô mỡ! Đó là lý do mô hình Deep Learning bị mù."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import random\n",
            "from PIL import Image\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "WSI_DIR = '/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/WSIs' \n",
            "\n",
            "if len(X_data) > 0 and len(misclassified_idx) > 0:\n",
            "    # Chọn ngẫu nhiên 1 bệnh nhân HER2 bị đoán sai\n",
            "    target_idx = misclassified_idx[0]\n",
            "    patient_id = valid_patients[target_idx] if 'valid_patients' in locals() else patient_ids[target_idx]\n",
            "    true_lbl = \"HER2\"\n",
            "    pred_lbl = ['LumA', 'LumB', 'Basal', 'HER2'][global_preds[target_idx]]\n",
            "    \n",
            "    print(f\"Khám nghiệm Bệnh nhân: {patient_id}\")\n",
            "    print(f\"Nhãn thực tế: {true_lbl} | Mô hình Naive đoán nhầm thành: {pred_lbl}\")\n",
            "    \n",
            "    patient_folder = os.path.join(WSI_DIR, patient_id)\n",
            "    if os.path.exists(patient_folder):\n",
            "        all_patches = [f for f in os.listdir(patient_folder) if f.endswith('.jpg') or f.endswith('.png')]\n",
            "        if len(all_patches) >= 16:\n",
            "            sample_patches = random.sample(all_patches, 16)\n",
            "            \n",
            "            fig, axes = plt.subplots(4, 4, figsize=(10, 10))\n",
            "            fig.suptitle(f\"Trực quan hóa 16 mảnh tế bào ngẫu nhiên của bệnh nhân {patient_id}\\n(Minh chứng cho sự hòa tan tín hiệu của Mean-Pooling)\", fontsize=14)\n",
            "            \n",
            "            for i, ax in enumerate(axes.flat):\n",
            "                img_path = os.path.join(patient_folder, sample_patches[i])\n",
            "                img = Image.open(img_path)\n",
            "                ax.imshow(img)\n",
            "                ax.axis('off')\n",
            "                \n",
            "            plt.tight_layout()\n",
            "            plt.subplots_adjust(top=0.9)\n",
            "            plt.show()\n",
            "        else:\n",
            "            print(f\"Bệnh nhân này có quá ít ảnh (<16) để hiển thị.\")\n",
            "    else:\n",
            "        print(f\"Không tìm thấy thư mục ảnh gốc tại {patient_folder}. Bỏ qua bước vẽ ảnh.\")\n",
            "elif len(X_data) > 0:\n",
            "    print(\"Tuyệt vời, không có ca HER2 nào bị đoán sai (hoặc không có dữ liệu để vẽ)....\")"
        ]
    }
]

nb['cells'].extend(new_cells)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Appended Visualization cells to Notebook 4 successfully.")
