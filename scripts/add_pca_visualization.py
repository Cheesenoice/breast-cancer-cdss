import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\03.1_Classification_Traditional_ML_Advanced.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    new_cells.append(cell)
    # Tìm cell code chạy PCA để chèn cell trực quan hóa ngay phía sau nó
    if cell['cell_type'] == 'code' and 'PCA(n_components=128' in ''.join(cell.get('source', [])):
        
        # Thêm Markdown mô tả
        md_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.1 Trực Quan Hóa Sức Mạnh Của PCA (Dành cho Slide Thuyết trình)\n",
                "Đoạn code dưới đây sẽ vẽ 2 biểu đồ chứng minh lý do tại sao PCA lại \"thần thánh\" đến vậy:\n",
                "1. **Biểu đồ Tích lũy Thông tin (Scree Plot):** Chứng minh bằng toán học rằng tại sao cắt bỏ từ 6144 chiều xuống 128 chiều mà không bị mất mát dữ liệu nghiêm trọng.\n",
                "2. **Biểu đồ Phân tán 2D (2D Projection):** Nén toàn bộ dữ liệu 6144 chiều xuống chỉ còn đúng 2 chiều (Trục X và Trục Y) để con người có thể nhìn thấy được bằng mắt thường xem các nhóm bệnh nhân có đang tụ lại thành cụm (Cluster) hay không."
            ]
        }
        
        # Thêm Code trực quan hóa
        code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if len(X_raw) > 0:\n",
                "    # Để vẽ biểu đồ Tích lũy, ta chạy thử PCA không giới hạn số chiều (nhưng tối đa là min(n_samples, n_features))\n",
                "    pca_full = PCA(random_state=42)\n",
                "    pca_full.fit(X_scaled)\n",
                "    \n",
                "    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)\n",
                "    \n",
                "    plt.figure(figsize=(16, 6))\n",
                "    \n",
                "    # BIỂU ĐỒ 1: TÍCH LŨY THÔNG TIN\n",
                "    plt.subplot(1, 2, 1)\n",
                "    plt.plot(cumulative_variance, color='blue', linewidth=2)\n",
                "    plt.axvline(x=128, color='red', linestyle='--', label='Ngưỡng nén 128 chiều')\n",
                "    plt.axhline(y=cumulative_variance[128] if len(cumulative_variance)>128 else 1.0, color='green', linestyle=':', label=f'Giữ lại ~{cumulative_variance[128]*100:.1f}% thông tin')\n",
                "    plt.title('Hiệu quả nén của PCA (Dimensionality vs Variance)', fontsize=14)\n",
                "    plt.xlabel('Số lượng chiều (Principal Components)')\n",
                "    plt.ylabel('Tỷ lệ thông tin được giữ lại (Cumulative Variance)')\n",
                "    plt.legend()\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    \n",
                "    # BIỂU ĐỒ 2: GÓC NHÌN 2D CỦA CON NGƯỜI\n",
                "    plt.subplot(1, 2, 2)\n",
                "    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_baseline, cmap='Set1', alpha=0.7, edgecolors='k')\n",
                "    \n",
                "    # Tạo chú thích cho 4 nhãn ung thư\n",
                "    legend_labels = {0: 'LumA', 1: 'LumB', 2: 'Basal', 3: 'HER2'}\n",
                "    handles, _ = scatter.legend_elements()\n",
                "    plt.legend(handles, [legend_labels[i] for i in range(4)], title=\"Phân nhóm\")\n",
                "    \n",
                "    plt.title('Bản đồ Phân tán 2D sau khi nén bằng PCA', fontsize=14)\n",
                "    plt.xlabel('Thành phần chính 1 (PC1)')\n",
                "    plt.ylabel('Thành phần chính 2 (PC2)')\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        }
        
        new_cells.extend([md_cell, code_cell])

nb['cells'] = new_cells

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Added PCA visualization cells successfully.")
