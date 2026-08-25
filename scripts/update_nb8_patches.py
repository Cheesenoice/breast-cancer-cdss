import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\08_Explainable_AI_Multimodal.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Sửa lỗi thiếu lifelines
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '!pip install' in ''.join(cell.get('source', [])):
        cell['source'] = ["!pip install nystrom-attention captum lifelines"]

# 2. Thêm module hiển thị ảnh Patches thật
real_image_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2.5 Hiển thị Ảnh Tế bào thực tế (Real Image Patches)\n",
            "<div class=\"alert alert-info\">\n",
            "Bản đồ nhiệt ở trên cho thấy AI chỉ tập trung vào một vài mảnh vỡ (Patches). Đoạn code dưới đây sẽ <strong>truy xuất trực tiếp các mảnh ảnh gốc (.png/.jpg)</strong> tương ứng với các tọa độ rực đỏ đó để Bác sĩ có thể xem tận mắt hình thái tế bào ung thư mà AI đã phát hiện!\n",
            "</div>"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from PIL import Image\n",
            "\n",
            "# TODO: Bạn hãy trỏ đường dẫn này tới thư mục chứa ảnh Patches gốc trên Kaggle\n",
            "# Ví dụ: PATCH_DIR = '/kaggle/input/tcga-brca-patches/tumor_patches'\n",
            "PATCH_DIR = '/kaggle/input/TÊN_DATASET_ẢNH_CỦA_BẠN'\n",
            "\n",
            "patient_patch_dir = os.path.join(PATCH_DIR, demo_pid)\n",
            "\n",
            "if os.path.exists(patient_patch_dir):\n",
            "    # Giả định các file ảnh được sắp xếp theo Alpha-bê (Giống hệt lúc trích xuất đặc trưng ResNet50)\n",
            "    all_patches = sorted(os.listdir(patient_patch_dir))\n",
            "    \n",
            "    # Lấy ra Top 5 mảnh quan trọng nhất từ mảng patch_importance\n",
            "    top_5_indices = np.argsort(patch_importance)[::-1][:5]\n",
            "    \n",
            "    plt.figure(figsize=(20, 4))\n",
            "    for i, idx in enumerate(top_5_indices):\n",
            "        if idx < len(all_patches):\n",
            "            img_path = os.path.join(patient_patch_dir, all_patches[idx])\n",
            "            img = Image.open(img_path)\n",
            "            plt.subplot(1, 5, i + 1)\n",
            "            plt.imshow(img)\n",
            "            plt.title(f\"Rank #{i+1}\\nScore: {patch_importance[idx]:.4f}\", fontsize=12, fontweight='bold', color='red')\n",
            "            plt.axis('off')\n",
            "    plt.suptitle(f\"Top 5 Mảnh Tế Bào Ác Tính Nhất - Bệnh nhân {demo_pid}\", fontsize=16, fontweight='bold')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(f\"⚠️ Không tìm thấy thư mục ảnh gốc: {patient_patch_dir}\")\n",
            "    print(\"Vui lòng Add Dataset chứa ảnh Patches gốc vào Kaggle, sau đó cập nhật lại biến PATCH_DIR để hiển thị!\")\n"
        ]
    }
]

# Tìm vị trí để chèn (Sau cell Heatmap)
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'Bản Đồ Nhiệt Khối U' in ''.join(cell.get('source', [])):
        insert_idx = i + 1
        break

if insert_idx != -1:
    for c in reversed(real_image_cells):
        nb['cells'].insert(insert_idx, c)
else:
    nb['cells'].extend(real_image_cells)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Đã fix lỗi lifelines và chèn cell hiển thị ảnh Patches thực tế.")
