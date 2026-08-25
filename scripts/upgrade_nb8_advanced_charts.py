import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\08_Explainable_AI_Multimodal.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

advanced_charts = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Hồ Sơ Bất Thường Gen (Genomic Anomaly Radar)\n",
            "<div class=\"alert alert-warning\">\n",
            "Mỗi bệnh nhân là một cá thể độc bản. Biểu đồ Radar (Mạng nhện) dưới đây so sánh Mức độ biểu hiện (Expression Level) của Top 5 Gen đột biến nguy hiểm nhất của bệnh nhân này so với <strong>Mức trung bình bình thường</strong> của cả bệnh viện. Bác sĩ sẽ biết ngay gen nào đang \"nổi loạn\" mạnh nhất.\n",
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
            "# Lấy 5 Gen mạnh nhất (Ở cuối mảng top_genes vì đã được sắp xếp tăng dần theo độ quan trọng)\n",
            "categories = top_genes[-5:].tolist()\n",
            "N = len(categories)\n",
            "patient_expr = rna_500_scaled.loc[demo_pid, categories].values.flatten().tolist()\n",
            "pop_expr = rna_500_scaled[categories].mean().values.flatten().tolist()\n",
            "\n",
            "# Nối điểm cuối với điểm đầu để khép kín hình mạng nhện\n",
            "patient_expr += patient_expr[:1]\n",
            "pop_expr += pop_expr[:1]\n",
            "angles = [n / float(N) * 2 * math.pi for n in range(N)]\n",
            "angles += angles[:1]\n",
            "\n",
            "plt.figure(figsize=(7,7))\n",
            "ax = plt.subplot(111, polar=True)\n",
            "plt.xticks(angles[:-1], categories, color='black', size=14, fontweight='bold')\n",
            "ax.plot(angles, patient_expr, linewidth=3, linestyle='solid', color='purple', label=f'Bệnh nhân {demo_pid}')\n",
            "ax.fill(angles, patient_expr, 'purple', alpha=0.2)\n",
            "ax.plot(angles, pop_expr, linewidth=2, linestyle='dashed', color='grey', label='Trung bình Quần thể (Base)')\n",
            "ax.fill(angles, pop_expr, 'grey', alpha=0.1)\n",
            "plt.title('Hồ sơ Đột biến Gen Cá nhân hóa (Radar Chart)', size=16, fontweight='bold', y=1.1)\n",
            "plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Tọa Độ Bệnh Nhân (Patient t-SNE Manifold)\n",
            "<div class=\"alert alert-info\">\n",
            "Dữ liệu của 882 bệnh nhân tạo thành một Vũ trụ Y tế (Medical Manifold) siêu phức tạp. Thuật toán t-SNE nén không gian 1024 chiều xuống còn 2 chiều. Ngôi sao <strong>ĐỎ KHỔNG LỒ</strong> chính là bệnh nhân của chúng ta. Bác sĩ sẽ biết ca bệnh này có phải là \"Ca điển hình\" (Nằm ở tâm cụm) hay là \"Ca hiếm\" (Nằm ở vùng biên giới).\n",
            "</div>"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.manifold import TSNE\n",
            "\n",
            "print(\"Đang định vị bệnh nhân trên Radar Không gian t-SNE...\")\n",
            "# Dùng fusion_pca (16D) đã tính ở Cell Tiên lượng Sinh tồn để t-SNE chạy siêu nhanh\n",
            "tsne = TSNE(n_components=2, random_state=42, perplexity=30)\n",
            "fusion_tsne = tsne.fit_transform(fusion_pca) \n",
            "\n",
            "tsne_df = pd.DataFrame(fusion_tsne, columns=['tsne_1', 'tsne_2'], index=valid_pids)\n",
            "tsne_df['label'] = df_clin['pam50_subtype'].values\n",
            "\n",
            "plt.figure(figsize=(12, 8))\n",
            "sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_df, palette='Set2', s=50, alpha=0.5)\n",
            "patient_coords = tsne_df.loc[demo_pid]\n",
            "\n",
            "# Đóng dấu Bệnh nhân hiện tại bằng Ngôi sao Đỏ\n",
            "plt.scatter(patient_coords['tsne_1'], patient_coords['tsne_2'], color='red', marker='*', s=1500, edgecolor='black', label=f'⭐ BỆNH NHÂN NÀY ({demo_pid})', zorder=10)\n",
            "plt.title('Định vị Bệnh nhân trong Quần thể Ung thư Vú (t-SNE Projection)', fontsize=16, fontweight='bold')\n",
            "plt.legend(title='Phân nhóm PAM50', fontsize=12)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Thước Đo Rủi Ro Tử Vong (Hazard Risk Density)\n",
            "<div class=\"alert alert-danger\">\n",
            "Biểu đồ cuối cùng trả lời câu hỏi sinh tử: <strong>\"Bệnh nhân này nguy kịch cỡ nào?\"</strong>.<br>\n",
            "Đỉnh núi màu đỏ thể hiện phân bố Điểm Rủi Ro (Hazard Score) của toàn bệnh viện. Vạch kẻ đứt màu đen chính là \"Bản án\" dành cho bệnh nhân này.\n",
            "</div>"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Khai thác mô hình CoxPH (cph) đã huấn luyện ngầm ở phần Sinh Tồn\n",
            "all_risks = cph.predict_partial_hazard(surv_df)\n",
            "patient_risk = all_risks.loc[demo_pid]\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.kdeplot(all_risks.values, fill=True, color='crimson', alpha=0.4)\n",
            "plt.axvline(x=patient_risk, color='black', linestyle='--', linewidth=4, label=f'Cảnh báo Rủi ro của Bệnh nhân: {patient_risk:.2f}')\n",
            "\n",
            "plt.title('Thước đo Cảnh báo Tử vong (Hazard Score Density)', fontsize=16, fontweight='bold')\n",
            "plt.xlabel('Điểm Rủi ro Tử vong (Hazard Score)')\n",
            "plt.ylabel('Mật độ (Density)')\n",
            "plt.legend(fontsize=12)\n",
            "plt.grid(True, linestyle='--', alpha=0.3)\n",
            "plt.show()"
        ]
    }
]

nb['cells'].extend(advanced_charts)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Đã bơm thêm 3 biểu đồ Advanced XAI (Radar, t-SNE, Risk Density) vào cuối Dashboard.")
