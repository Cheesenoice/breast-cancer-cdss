import json
import os

def add_save_weights_to_notebook(path, prefix):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'best_fold_preds = preds' in source:
                new_source = []
                lines = source.split('\n')
                for line in lines:
                    new_source.append(line)
                    if 'best_fold_labels = tgts' in line:
                        indent = line[:len(line) - len(line.lstrip())]
                        new_source.append(f"{indent}torch.save(model.state_dict(), f'/kaggle/working/{prefix}_fold{{fold+1}}_best.pth')")
                cell['source'] = '\n'.join(new_source)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

update_nb1_path = r'C:\Users\huynh\Desktop\breast cancer\06.1_Multimodal_WSI_Clinical.ipynb'
update_nb2_path = r'C:\Users\huynh\Desktop\breast cancer\06.2_Multimodal_WSI_Genomics.ipynb'

if os.path.exists(update_nb1_path):
    add_save_weights_to_notebook(update_nb1_path, 'multimodal_clinical')
if os.path.exists(update_nb2_path):
    add_save_weights_to_notebook(update_nb2_path, 'multimodal_genomics')

print("Đã chèn lệnh lưu trọng số thành công.")
