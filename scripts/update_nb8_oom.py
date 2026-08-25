import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\08_Explainable_AI_Multimodal.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        
        # 1. Cập nhật lại đường dẫn đúng
        if 'WEIGHTS_PATH =' in source:
            new_source = []
            for line in source.split('\n'):
                if 'WEIGHTS_PATH =' in line:
                    new_source.append("    WEIGHTS_PATH = '/kaggle/input/datasets/trihuynhviprovcl/tcga-brca-multimodal-genomics-weights/multimodal_genomics_fold1_best.pth'")
                else:
                    new_source.append(line)
            cell['source'] = '\n'.join(new_source)
            
        # 2. Sửa lỗi OOM của Captum
        if 'ig.attribute(' in source:
            new_source = source.replace(
                "attributions, delta = ig.attribute(inputs=(img_tensor, gen_tensor), target=int(true_label), return_convergence_delta=True)",
                "attributions, delta = ig.attribute(inputs=(img_tensor, gen_tensor), target=int(true_label), internal_batch_size=1, n_steps=50, return_convergence_delta=True)"
            )
            cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Đã sửa lỗi CUDA OOM và cập nhật đường dẫn thành công.")
