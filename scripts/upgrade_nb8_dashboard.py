import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\08_Explainable_AI_Multimodal.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

dashboard_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# TỔNG TRẠM Y KHOA (AI CLINICAL DASHBOARD)\n",
            "<div class=\"alert alert-success\">\n",
            "Mọi phân tích từ đầu đến giờ sẽ được hội tụ tại đây. Bảng điều khiển này cung cấp cho Bác sĩ một cái nhìn toàn cảnh (360 độ) về bệnh nhân <code>demo_pid</code>: Từ thông tin cá nhân, tỷ lệ chẩn đoán, vùng không gian khối u (Heatmap giả lập), cho đến tiên lượng tuổi thọ!\n",
            "</div>"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import display, HTML\n",
            "\n",
            "patient_row = df_clin.loc[demo_pid]\n",
            "real_months = patient_row['OS_MONTHS']\n",
            "real_status_text = \"ĐÃ TỬ VONG\" if ('DECEASED' in str(patient_row['OS_STATUS']) or '1:' in str(patient_row['OS_STATUS'])) else \"CÒN SỐNG\"\n",
            "age = patient_row['AGE']\n",
            "stage = patient_row.get('AJCC_PATHOLOGIC_TUMOR_STAGE', 'Unknown')\n",
            "\n",
            "html_content = f\"\"\"\n",
            "<div style=\"background-color:#f8f9fa; padding:20px; border-radius:10px; border-left: 6px solid #007bff; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);\">\n",
            "    <h2 style=\"color:#007bff; margin-top:0;\">HỒ SƠ BỆNH ÁN: {demo_pid}</h2>\n",
            "    <ul style=\"font-size: 16px; line-height: 1.8;\">\n",
            "        <li><b>Tuổi:</b> {age}</li>\n",
            "        <li><b>Giai đoạn Khối u:</b> {stage}</li>\n",
            "        <li><b>Nhóm bệnh lý thực tế:</b> <span style=\"color:red; font-weight:bold;\">{subtype_name}</span></li>\n",
            "        <li><b>Tình trạng Lâm sàng:</b> {real_status_text} (Thời gian theo dõi: {real_months} tháng)</li>\n",
            "    </ul>\n",
            "</div>\n",
            "\"\"\"\n",
            "display(HTML(html_content))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. AI Hội Chẩn (Multimodal Diagnosis)\n",
            "Mô hình Đa phương thức (Ảnh + Gen) dự đoán khả năng mắc 4 nhóm bệnh. (Softmax Probabilities)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch.nn.functional as F\n",
            "\n",
            "# Tính xác suất từ Logits\n",
            "model.eval()\n",
            "with torch.no_grad():\n",
            "    logits = model(img_tensor, gen_tensor)\n",
            "    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy() * 100\n",
            "\n",
            "labels = ['LumA', 'LumB', 'Basal', 'HER2']\n",
            "colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']\n",
            "explode = [0.1 if i == np.argmax(probs) else 0 for i in range(4)]\n",
            "\n",
            "plt.figure(figsize=(7, 7))\n",
            "plt.pie(probs, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',\n",
            "        shadow=True, startangle=140, textprops={'fontsize': 14, 'fontweight': 'bold'})\n",
            "plt.title(\"Tỷ lệ Dự đoán Đa phương thức (AI Confidence)\", fontsize=16, fontweight='bold')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Bản Đồ Nhiệt Cắt Lớp (Digital Slide Heatmap)\n",
            "<div class=\"alert alert-warning\">\n",
            "Vì giới hạn lưu trữ, ta không còn giữ ảnh màu thực tế của bệnh nhân. AI sẽ tái tạo một Bản đồ Nhiệt dạng lưới (Grid 2D) đại diện cho kính hiển vi. Mỗi ô vuông là 1 mảng tế bào. Các ô màu <strong>Đỏ/Cam</strong> rực lên chính là các tọa độ <strong>lõi ung thư</strong> mà AI đã soi chiếu ra!\n",
            "</div>"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import math\n",
            "\n",
            "n_patches = len(patch_importance)\n",
            "side = math.ceil(math.sqrt(n_patches))\n",
            "padded_len = side * side\n",
            "\n",
            "# Bơm thêm NaN vào cho đủ hình vuông\n",
            "padded_importance = np.pad(patch_importance, (0, padded_len - n_patches), constant_values=np.nan)\n",
            "grid = padded_importance.reshape((side, side))\n",
            "\n",
            "plt.figure(figsize=(10, 8))\n",
            "# Dùng cmap 'coolwarm' để vùng chú ý thấp màu xanh, vùng khối u màu đỏ rực\n",
            "sns.heatmap(grid, cmap='coolwarm', cbar_kws={'label': 'Mức độ Chú ý (Attention Score)'}, \n",
            "            xticklabels=False, yticklabels=False, mask=np.isnan(grid))\n",
            "plt.title(f\"Bản Đồ Nhiệt Khối U (Mô phỏng trên {n_patches} mảnh tế bào)\", fontsize=16, fontweight='bold')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Tiên Lượng Sinh Tồn Cá Nhân Hóa (Survival Prognosis)\n",
            "<div class=\"alert alert-danger\">\n",
            "Chạy ngầm thuật toán CoxPH để vẽ Quỹ đạo Sinh tồn dự kiến của bệnh nhân này. AI Đa phương thức sẽ đối chiếu quỹ đạo này với đường kẻ vạch tử thần ngoài đời thực!\n",
            "</div>"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.decomposition import PCA\n",
            "from lifelines import CoxPHFitter\n",
            "\n",
            "print(\"Đang nạp nhanh Dữ liệu Tiên lượng ngầm (Mất khoảng 30s)...\")\n",
            "\n",
            "def parse_stage(s):\n",
            "    s = str(s).upper()\n",
            "    if 'IV' in s: return 4\n",
            "    if 'III' in s: return 3\n",
            "    if 'II' in s: return 2\n",
            "    if 'I' in s: return 1\n",
            "    return 1\n",
            "\n",
            "# Trích xuất Fusion Features (Cấp tốc)\n",
            "fusion_features = []\n",
            "with torch.no_grad():\n",
            "    for pid in valid_pids:\n",
            "        img = torch.load(os.path.join(PT_DIR, f\"{pid}.pt\"), map_location=device).unsqueeze(0)\n",
            "        gen = torch.tensor(rna_500_scaled.loc[pid].values, dtype=torch.float32).unsqueeze(0).to(device)\n",
            "        # Trích xuất lớp ẩn trước Classifier (Layer [0])\n",
            "        v_img = model.vision_net(img)\n",
            "        v_gen = model.genomics_net(gen)\n",
            "        v_fusion = torch.cat((v_img, v_gen), dim=1)\n",
            "        fusion_features.append(v_fusion.cpu().numpy()[0])\n",
            "\n",
            "fusion_features = np.array(fusion_features)\n",
            "pca = PCA(n_components=16)\n",
            "fusion_pca = pca.fit_transform(fusion_features)\n",
            "\n",
            "surv_df = pd.DataFrame(fusion_pca, columns=[f'PC_{i}' for i in range(16)], index=valid_pids)\n",
            "surv_df['OS_MONTHS'] = df_clin['OS_MONTHS']\n",
            "surv_df['OS_STATUS'] = df_clin['OS_STATUS'].apply(lambda x: 1 if 'DECEASED' in str(x) or '1:' in str(x) else 0)\n",
            "surv_df['AGE'] = df_clin['AGE']\n",
            "surv_df['STAGE'] = df_clin['AJCC_PATHOLOGIC_TUMOR_STAGE'].apply(parse_stage)\n",
            "\n",
            "cph = CoxPHFitter(penalizer=0.1)\n",
            "cph.fit(surv_df, duration_col='OS_MONTHS', event_col='OS_STATUS')\n",
            "\n",
            "patient_row_surv = surv_df.loc[[demo_pid]]\n",
            "patient_survival = cph.predict_survival_function(patient_row_surv)\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.plot(patient_survival.index, patient_survival.iloc[:, 0], color='purple', lw=4, label='Dự đoán Quỹ Đạo Sống')\n",
            "plt.fill_between(patient_survival.index, 0, patient_survival.iloc[:, 0], color='purple', alpha=0.1)\n",
            "\n",
            "real_status_num = patient_row_surv['OS_STATUS'].values[0]\n",
            "if real_status_num == 1:\n",
            "    plt.axvline(x=real_months, color='red', linestyle='--', linewidth=2, label='Thời điểm Tử vong Thực tế')\n",
            "    plt.scatter(real_months, patient_survival.loc[real_months].values[0] if real_months in patient_survival.index else 0.5, color='red', s=150, zorder=5)\n",
            "else:\n",
            "    plt.axvline(x=real_months, color='green', linestyle='--', linewidth=2, label='Lần Theo Dõi Cuối (Còn Sống)')\n",
            "\n",
            "plt.title(f\"Quỹ Đạo Sinh Tồn - Bệnh nhân {demo_pid}\", fontsize=16, fontweight='bold')\n",
            "plt.xlabel(\"Thời gian (Tháng)\", fontsize=12)\n",
            "plt.ylabel(\"Xác suất Sống sót\", fontsize=12)\n",
            "plt.legend(fontsize=12)\n",
            "plt.grid(True, linestyle='--', alpha=0.6)\n",
            "plt.show()"
        ]
    }
]

nb['cells'].extend(dashboard_cells)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Đã tích hợp AI Clinical Dashboard vào phần cuối Notebook 8 thành công.")
