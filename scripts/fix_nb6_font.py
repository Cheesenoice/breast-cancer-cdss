import json
import os

path1 = r'C:\Users\huynh\Desktop\breast cancer\06.0_Multimodal_EDA_and_Genomics_Processing.ipynb'
path2 = r'C:\Users\huynh\Desktop\breast cancer\nb-w-output\06-0-multimodal-eda-and-genomics-processing.ipynb'

def fix_font_and_export(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'import matplotlib.pyplot as plt' in source:
                new_source = []
                for line in source.split('\n'):
                    new_source.append(line)
                    if "plt.rcParams['axes.labelsize'] = 12" in line:
                        # Thêm cấu hình font tiếng Việt
                        new_source.append("\n# Cấu hình hiển thị chuẩn Tiếng Việt cho Matplotlib")
                        new_source.append("plt.rcParams['font.family'] = 'sans-serif'")
                        new_source.append("plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans', 'Liberation Sans']")
                        new_source.append("plt.rcParams['axes.unicode_minus'] = False")
                cell['source'] = '\n'.join(new_source)
                
            # Cũng thêm lệnh xuất CSV vào cell cuối cùng nếu cần thiết (optional, theo đoạn đối thoại trước)
            if 'plt.show()' in source and 'tsne_gen = TSNE' in source: # Cell vẽ t-SNE cuối cùng
                pass
                
    # Thêm 1 cell xuất CSV vào cuối notebook như đã hứa
    export_cell = {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# 7. Xuất ma trận Gen đã tiền xử lý thành file gọn nhẹ (~3MB)\n",
      "# File này sẽ được Notebook 6.2 tải trực tiếp thay vì phải đọc lại file 154MB thô!\n",
      "out_csv = '/kaggle/working/tcga_brca_rna_500_scaled.csv'\n",
      "rna_500_scaled.to_csv(out_csv)\n",
      "print(f\"✓ Đã xuất file ma trận 500 gen sạch thành công tại: {out_csv}\")"
     ]
    }
    
    # Check if the export cell already exists
    has_export = any('tcga_brca_rna_500_scaled.csv' in "".join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
    if not has_export:
        # Insert before the last markdown cell
        nb['cells'].insert(-1, export_cell)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_font_and_export(path1)
fix_font_and_export(path2)
print("Đã sửa lỗi font tiếng Việt và thêm code xuất CSV.")
