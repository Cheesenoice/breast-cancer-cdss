import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\07_Survival_Analysis_Multimodal.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'WEIGHTS_PATH =' in source:
            new_source = source.replace(
                "WEIGHTS_PATH = '/kaggle/input/TÊN_DATASET_CỦA_BẠN/multimodal_genomics_fold1_best.pth'",
                "WEIGHTS_PATH = '/kaggle/input/tcga-brca-multimodal-genomics-weights/multimodal_genomics_fold1_best.pth'"
            )
            cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Đã cập nhật đường dẫn trọng số thành công.")
