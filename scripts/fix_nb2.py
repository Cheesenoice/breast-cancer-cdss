import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\02_CNN_Feature_Extraction_and_Baseline.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'train_test_split' in ''.join(cell['source']):
        source_str = ''.join(cell['source'])
        if 'stratify=y_baseline' in source_str and 'min_class_count' not in source_str:
            new_source = []
            for line in cell['source']:
                if 'train_test_split(X_baseline, y_baseline' in line:
                    new_source.extend([
                        "    # Tự động tắt stratify nếu tập dữ liệu nháp (limit) có class < 2 mẫu\n",
                        "    from collections import Counter\n",
                        "    min_class_count = min(Counter(y_baseline).values())\n",
                        "    stratify_param = y_baseline if min_class_count >= 2 else None\n",
                        "    \n",
                        "    X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=0.2, stratify=stratify_param, random_state=42)\n"
                    ])
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed NB2 successfully.")
