import json
import os

path1 = r'C:\Users\huynh\Desktop\breast cancer\06.0_Multimodal_EDA_and_Genomics_Processing.ipynb'
path2 = r'C:\Users\huynh\Desktop\breast cancer\nb-w-output\06-0-multimodal-eda-and-genomics-processing.ipynb'

def fix_csv_export(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'out_csv = \'/kaggle/working/tcga_brca_rna_500_scaled.csv\'' in source:
                new_source = [
                    "# 7. Chuẩn hóa và Xuất ma trận Gen đã tiền xử lý thành file gọn nhẹ (~3MB)\n",
                    "# File này sẽ được Notebook 6.2 tải trực tiếp thay vì phải đọc lại file 154MB thô!\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "import pandas as pd\n",
                    "\n",
                    "# Chuẩn hóa Z-score toàn bộ ma trận 500 gen\n",
                    "scaler = StandardScaler()\n",
                    "rna_500_scaled = pd.DataFrame(scaler.fit_transform(rna_500), index=rna_500.index, columns=rna_500.columns)\n",
                    "\n",
                    "out_csv = '/kaggle/working/tcga_brca_rna_500_scaled.csv'\n",
                    "rna_500_scaled.to_csv(out_csv)\n",
                    "print(f\"✓ Đã xuất file ma trận 500 gen sạch thành công tại: {out_csv}\")"
                ]
                cell['source'] = new_source

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_csv_export(path1)
fix_csv_export(path2)
print("Đã sửa lỗi NameError rna_500_scaled ở cell cuối cùng.")
