import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\01_TCGA_EDA_and_Clinical_Processing.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_theory = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 🔬 Lý Thuyết Hình Thái Học (Morphology) Của 4 Nhóm Phân Tử\n",
    "\n",
    "Để hiểu cách mô hình AI học, chúng ta cần đóng vai trò là một bác sĩ giải phẫu bệnh thực thụ. 4 nhóm phân tử này không chỉ khác biệt ở cấp độ gen (DNA/RNA) mà còn bộc lộ những khác biệt vi thể trên ảnh WSI:\n",
    "\n",
    "| Phân nhóm | Mức độ ác tính (Grade) | Đặc điểm Hình thái tế bào (Morphological Features) | AI (ResNet + TransMIL) sẽ học như thế nào? |\n",
    "|---|---|---|---|\n",
    "| **Luminal A (LumA)** | Thấp (Low-grade) | Tế bào phân hóa tốt, khá giống mô tuyến vú bình thường. Hình thái nhân nhỏ, đều đặn, tỷ lệ phân bào (mitosis) cực kỳ thấp. Mô đệm (stroma) có thể phản ứng sợi hóa nhẹ. | Khá dễ nhận biết. AI sẽ tìm kiếm các vùng mô có cấu trúc ống tuyến (tubules) còn giữ được hình dạng, mật độ tế bào không quá dày đặc, và hạt nhân đồng đều. |\n",
    "| **Luminal B (LumB)** | Trung bình - Cao | Tế bào phân hóa kém hơn LumA. Nhân tế bào to hơn, mất cân xứng (pleomorphism) rõ rệt. **Đặc điểm nhận diện chính:** Tỷ lệ phân bào cao (nhiều tế bào đang nhân đôi). | AI sẽ tập trung vào sự đa hình của nhân tế bào và mật độ tế bào dày đặc hơn. Khó khăn: Dễ nhầm lẫn với HER2 nếu khối u có cấu trúc tương tự. |\n",
    "| **HER2-enriched** | Cao (High-grade) | Tế bào rất to, nhân thô và lớn (high nuclear pleomorphism). Có nhiều vùng hoại tử (necrosis) ở trung tâm. Tỷ lệ phân bào rất cao. | Khó khăn lớn cho AI: Hình thái của nhóm này dao động rất mạnh, thường chồng chéo với LumB. AI sẽ tìm kiếm các mảng tế bào to, viền tế bào không rõ ràng và vùng mô hoại tử, nhưng cần Weighted Loss để ép mạng chú ý kỹ hơn. |\n",
    "| **Basal-like** | Rất Cao (Highest-grade) | Cấu trúc khối u dày đặc (solid sheets), không còn hình dáng ống tuyến. **Đặc trưng nhận diện:** Bờ khối u thường là \"bờ đẩy\" (pushing border) rất rõ nét, xung quanh có rất nhiều tế bào lympho xâm nhập (Lymphocytic infiltration). | Dễ nhận biết nhất trong các nhóm ác tính cao. Chú ý của AI (Attention) thường sẽ tập trung mạnh vào các dải tế bào lympho vây quanh khối u và cấu trúc mảng đặc (solid pattern) của tế bào ung thư. |\n",
    "\n",
    "**💡 Kết luận:** Việc kết hợp **ResNet50** (Giỏi nhận diện các texture nhỏ như hạt nhân, viền tế bào) và **TransMIL** (Giỏi nhìn tổng quan cấu trúc không gian như sự phân bố của lympho và mô đệm) là sự kết hợp hoàn hảo để giải mã các đặc điểm hình thái học phức tạp này."
   ]
}

new_code_source = [
    "def show_4_subtypes_patches(wsi_dir, df, matched_list, patches_per_patient=4):\n",
    "    if not matched_list or not os.path.exists(wsi_dir):\n",
    "        print(\"Không thể trực quan hóa do chưa có ảnh hoặc chưa khớp ID.\")\n",
    "        return\n",
    "        \n",
    "    subtypes = ['BRCA_LumA', 'BRCA_LumB', 'BRCA_Her2', 'BRCA_Basal']\n",
    "    selected_pids = []\n",
    "    \n",
    "    # Cố gắng lấy 1 bệnh nhân đại diện cho mỗi Subtype\n",
    "    for st in subtypes:\n",
    "        st_patients = df[(df['pam50_subtype'] == st) & (df['patientId'].isin(matched_list))]['patientId'].tolist()\n",
    "        if st_patients:\n",
    "            selected_pids.append((st, random.choice(st_patients)))\n",
    "            \n",
    "    if not selected_pids:\n",
    "        print(\"Không tìm thấy bệnh nhân nào có nhãn phù hợp để hiển thị.\")\n",
    "        return\n",
    "        \n",
    "    fig, axes = plt.subplots(len(selected_pids), patches_per_patient, figsize=(15, 3.5 * len(selected_pids)))\n",
    "    if len(selected_pids) == 1:\n",
    "        axes = [axes]\n",
    "        \n",
    "    for i, (subtype, pid) in enumerate(selected_pids):\n",
    "        p_dir = os.path.join(wsi_dir, pid)\n",
    "        patches = [p for p in os.listdir(p_dir) if p.endswith('.jpg')]\n",
    "        selected_patches = random.sample(patches, min(patches_per_patient, len(patches)))\n",
    "        \n",
    "        for j, patch_name in enumerate(selected_patches):\n",
    "            img_path = os.path.join(p_dir, patch_name)\n",
    "            try:\n",
    "                img = Image.open(img_path)\n",
    "                ax = axes[i][j] if patches_per_patient > 1 else axes[i]\n",
    "                ax.imshow(img)\n",
    "                ax.axis('off')\n",
    "                if j == 0:\n",
    "                    ax.set_title(f\"Subtype: {subtype}\\nID: {pid}\", fontsize=12, fontweight='bold', color='darkred', loc='left')\n",
    "            except:\n",
    "                pass\n",
    "                \n",
    "    plt.suptitle(\"Hình Thái Tế Bào Đại Diện Của 4 Nhóm Phân Tử\", fontsize=16, fontweight='bold', y=1.02)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "if 'matched_patients' in locals():\n",
    "    show_4_subtypes_patches(WSI_DIR, df, matched_patients)"
]

# Xóa cell theory cũ nếu chạy nhiều lần để tránh bị lặp
nb['cells'] = [cell for cell in nb['cells'] if '### 🔬 Lý Thuyết Hình Thái Học' not in ''.join(cell.get('source', []))]

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '### Trực Quan Hóa Tế Bào Bệnh Học' in ''.join(cell.get('source', [])):
        nb['cells'][i+1]['source'] = new_code_source
        nb['cells'].insert(i+2, markdown_theory)
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Notebook 1 successfully with 4 subtypes and theory.")
