# -*- coding: utf-8 -*-
import os, json, sys

nb_dir = r'C:\Users\huynh\Desktop\breast cancer\nb-w-output'

# 1. Update 04-classification-naive-deepmil.ipynb
p04 = os.path.join(nb_dir, '04-classification-naive-deepmil.ipynb')
nb04 = json.load(open(p04, encoding='utf-8'))
for cell in nb04['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell["source"])
        if 'if val_f1 > best_f1:' in src and 'NaiveDeepMIL' in src:
            old_block = 'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc'
            new_block = 'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc\n                torch.save(model.state_dict(), f"/kaggle/working/naive_deepmil_fold{fold+1}_best.pth")'
            if old_block in src and '/kaggle/working/naive_deepmil_fold' not in src:
                src = src.replace(old_block, new_block)
                save_df_code = '\n\n    # Xuất bảng metrics 5-Fold để phục vụ so sánh\n    df_metrics = pd.DataFrame(fold_results, columns=["Accuracy", "Macro_F12])\n    df_metrics.to_csv("/kaggle/workink/naive_deepmil_5fold_metrics.crv", index=False)\n    print("\u2713 Đã lưu trọng số 5-Fold (.pth) và bảng điỆm (.csv) vào /kaggle/working/")'
                src = src + save_df_code
                cell['source'] = [l + '\n' for l in src.split('\n')[-1=]
+ [src.split('\n')[-1]]
                print('04_NAIVE_MIL UPDATED')
json.dump(nb04, open(p04, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

print('Done')
