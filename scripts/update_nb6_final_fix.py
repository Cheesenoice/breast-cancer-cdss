import json
import os

def upgrade_and_fix_notebook(path, is_genomics=False):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Sửa lỗi BatchNorm1d -> LayerNorm để tránh crash khi DataParallel chia BatchSize=1
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'nn.BatchNorm1d(64)' in source:
                cell['source'] = source.replace('nn.BatchNorm1d(64)', 'nn.LayerNorm(64)')
            if 'nn.BatchNorm1d(256)' in source:
                cell['source'] = source.replace('nn.BatchNorm1d(256)', 'nn.LayerNorm(256)')
                
    # --- CHÈN MARKDOWN SLIDES ---
    
    slide_1 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Slide 1: Bức tường Giới hạn của Kính hiển vi\n",
            "<div class=\"alert alert-info\">\n",
            "<strong>Tại sao WSI thất bại ở F1 ~0.50?</strong><br>\n",
            "Mắt người hay AI (qua ảnh WSI) chỉ có thể nhìn thấy <strong>Hình thái học (Morphology)</strong> - tức là vỏ bề ngoài của tế bào. Nó rất giỏi phân biệt tế bào lành và tế bào ung thư. Tuy nhiên, 4 phân nhóm PAM50 (LumA, LumB, Basal, HER2) được quyết định bởi <strong>Đột biến Gen</strong> nằm sâu trong lõi tế bào. Đòi hỏi AI phải đoán đột biến gen chỉ bằng cách nhìn bề ngoài là một nhiệm vụ bất khả thi, dẫn đến hiện tượng chạm trần (Ceiling Limit).\n",
            "</div>"
        ]
    }
    
    slide_2 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Slide 2: Lý thuyết Late Fusion (Hợp nhất trễ)\n",
            "<div class=\"alert alert-success\">\n",
            "<strong>Giải pháp Đa phương thức (Multimodal):</strong><br>\n",
            "Thay vì trộn dữ liệu thô từ đầu (Early Fusion) gây nhiễu loạn, chúng ta để mỗi mạng Neural tự chắt lọc tinh hoa của mình:\n",
            "<ul>\n",
            "<li>Nhánh WSI: Ép 5000 mảnh tế bào thành 1 vector <code>512</code> chiều.</li>\n",
            "<li>Nhánh Hồ sơ: Ép dữ liệu số (Lâm sàng/Gen) thành 1 vector <code>512</code> chiều.</li>\n",
            "</ul>\n",
            "Sau đó, lớp Fusion Gate sẽ nối (<code>torch.cat</code>) 2 vector này thành siêu vector <code>1024</code> chiều. Ở đây, trí tuệ của Mắt và Gen hòa quyện hoàn hảo để chốt nhãn cuối cùng.\n",
            "</div>"
        ]
    }
    
    slide_3 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Slide 3: Kỹ thuật Variance Threshold (Bộ lọc Gen)\n",
            "<div class=\"alert alert-warning\">\n",
            "<strong>Curse of Dimensionality (Lời nguyền chiều dữ liệu):</strong><br>\n",
            "Bản đồ gen RNA-Seq có đến 20,500 gen. Nếu đưa tất cả vào mạng Neural với chỉ 882 bệnh nhân, mô hình sẽ nổ tung và Overfitting ngay lập tức. Chúng ta sử dụng <strong>Variance Threshold</strong> để tự động đo lường và loại bỏ các gen tĩnh, chỉ giữ lại đúng <strong>500 gen biến động mạnh nhất</strong> - những \"công tắc\" quyết định sự khác biệt sinh học giữa các phân nhóm ung thư.\n",
            "</div>"
        ]
    }
    
    slide_4 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Slide 4: Vũ khí Huấn luyện Tối thượng\n",
            "<div class=\"alert alert-danger\">\n",
            "<strong>Cấu hình SOTA để chống Overfitting:</strong>\n",
            "<ul>\n",
            "<li><strong>RAdam (Rectified Adam):</strong> Tự động hãm đà học ở những Epoch đầu, giúp mạng Transformer học cực kỳ điềm tĩnh.</li>\n",
            "<li><strong>Gradient Accumulation = 8:</strong> Đọc lướt 8 bệnh nhân mới cập nhật tạ một lần, loại bỏ nhiễu giật cục.</li>\n",
            "<li><strong>Dual-GPU DataParallel:</strong> Ép 2 card T4 gánh 2 bệnh nhân cùng lúc, giảm một nửa thời gian train. Lỗi <code>BatchNorm1d</code> kinh điển của PyTorch khi chia batch=1 đã được khắc phục bằng <code>LayerNorm</code>.</li>\n",
            "</ul>\n",
            "</div>"
        ]
    }
    
    # Chèn các Slide vào đầu Notebook
    if is_genomics:
        nb['cells'].insert(1, slide_1)
        nb['cells'].insert(3, slide_3) # Chèn sau code block đầu
        nb['cells'].insert(5, slide_2)
        nb['cells'].insert(7, slide_4)
    else:
        nb['cells'].insert(1, slide_1)
        nb['cells'].insert(3, slide_2)
        nb['cells'].insert(5, slide_4)
        
    # --- THÊM T-SNE VISUALIZATION ---
    tsne_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.manifold import TSNE\n",
            "import numpy as np\n",
            "\n",
            "# Trực quan hóa Không gian Fusion 1024 chiều bằng t-SNE\n",
            "if len(valid_pids) > 0:\n",
            "    print(\"Đang nén Không gian Hợp nhất 1024 chiều xuống 2D bằng t-SNE...\")\n",
            "    model.eval()\n",
            "    fusion_vectors = []\n",
            "    fusion_labels = []\n",
            "    \n",
            "    with torch.no_grad():\n",
            "        for img, tab, label, _ in test_loader:\n",
            "            img, tab = img.to(device), tab.to(device)\n",
            "            \n",
            "            # Trích xuất tính năng từ 2 nhánh\n",
            "            v_img = model.module.vision_net(img) if hasattr(model, 'module') else model.vision_net(img)\n",
            "            if hasattr(model, 'module') and hasattr(model.module, 'genomics_net'):\n",
            "                v_tab = model.module.genomics_net(tab)\n",
            "            elif hasattr(model, 'module') and hasattr(model.module, 'tabular_net'):\n",
            "                v_tab = model.module.tabular_net(tab)\n",
            "            elif hasattr(model, 'genomics_net'):\n",
            "                v_tab = model.genomics_net(tab)\n",
            "            else:\n",
            "                v_tab = model.tabular_net(tab)\n",
            "                \n",
            "            # Gộp lại thành 1024\n",
            "            v_fusion = torch.cat((v_img, v_tab), dim=1)\n",
            "            \n",
            "            fusion_vectors.extend(v_fusion.cpu().numpy())\n",
            "            fusion_labels.extend(label.numpy())\n",
            "\n",
            "    fusion_vectors = np.array(fusion_vectors)\n",
            "    tsne = TSNE(n_components=2, random_state=42, perplexity=15)\n",
            "    tsne_results = tsne.fit_transform(fusion_vectors)\n",
            "    \n",
            "    plt.figure(figsize=(10, 8))\n",
            "    sns.scatterplot(\n",
            "        x=tsne_results[:, 0], y=tsne_results[:, 1],\n",
            "        hue=[['LumA', 'LumB', 'Basal', 'HER2'][l] for l in fusion_labels],\n",
            "        palette=sns.color_palette(\"hsv\", 4),\n",
            "        legend=\"full\",\n",
            "        alpha=0.8, s=100\n",
            "    )\n",
            "    plt.title('t-SNE Visualization: Không Gian Đa Phương Thức (Late Fusion)', fontsize=16, fontweight='bold')\n",
            "    plt.show()"
        ]
    }
    nb['cells'].append(tsne_cell)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

upgrade_and_fix_notebook(r'C:\Users\huynh\Desktop\breast cancer\06.1_Multimodal_WSI_Clinical.ipynb', is_genomics=False)
upgrade_and_fix_notebook(r'C:\Users\huynh\Desktop\breast cancer\06.2_Multimodal_WSI_Genomics.ipynb', is_genomics=True)
print("Sửa lỗi BatchNorm -> LayerNorm và nâng cấp Slide t-SNE thành công.")
