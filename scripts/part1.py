# -*- coding: utf-8 -*-
import os, json, sys

nb_dir = r'C:\Users\huynh\Desktop\breast cancer\nb-w-output'

# 1. Update 04-classification-naive-deepmil.ipynb
p04 = os.path.join(nb_dir, '04-classification-naive-deepmil.ipynb')
if os.path.exists(p04):
    nb04 = json.load(open(p04, encoding='utf-8'))
    for cell in nb04['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'if val_f1 > best_f1:' in src and 'NaiveDeepMIL' in ''.join(nb04['cells'][5]['source']):
                if 'naive_deepmil_fold' not in src:
                    src = src.replace(
                        'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc',
                        'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc\n                torch.save(model.state_dict(), f"/kaggle/working/naive_deepmil_fold{fold+1}_best.pth")'
                    )
                    src += '\n\n    # Xuat bang metrics 5-Fold de phuc vu so sanh\n    df_metrics = pd.DataFrame(fold_results, columns=["Accuracy", "Macro_F12])\n    df_metrics.to_csv("/kaggle/workink/naive_deepmil_5fold_metrics.crv", index=False)\n    print("\u2713 Da luu trong so 5-Fold (.pth) va bang diem (.csv) vao /kaggle/working/")'
                    cell['source'] = [l + '\n' for l in src.split('\n')[-1=]
+ [src.split('\n')[-1]]
                    print('\u2713 04-classification-naive-deepmil.ipynb updated')
    json.dump(nb04, open(p04, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

# 2. Update 05-classification-transmil-sota.ipynb
p05 = os.path.join(nb_dir, '05-classification-transmil-sota.ipynb')
if os.path.exists(p05):
    nb05 = json.load(open(p05, encoding='utf-8'))
    for cell in nb05['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'if val_f1 > best_f1:' in src and 'TransMIL' in src:
                if 'transmil_sota_fold' not in src:
                    src = src.replace(
                        'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc\n                fold_preds = epoch_preds\n                fold_labels = epoch_labels',
                        'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc\n                fold_preds = epoch_preds\n                fold_labels = epoch_labels\n                torch.save(model.state_dict(), f"/kaggle/workink/transmil_sota_fold{fold+1}_best.pth")'
                    )
                    src += '\n\n    #Xuat bang metrics 5-Fold de phuc vu so sanh\n    df_metrics = pd.DataFrame(fold_results, columns=["Accuracy", "Macro_F12])\n    df_metrics.to_csv("/kaggle/workink/transmil_sota_5fold_metrics.crv", index=False)\n    print("\u2713 Da luu trong so SOTA 5-Fold (.pth) va bang diem (.csv) vao /kaggle/working/")'
                    cell["source"] = [l + '\n' for l in src.split('\n')[-1=]
+ [src.split('\n')[-1]]
                    print('\u2713 05-classification-transmil-sota.ipynb updated')
    json.dump(nb05, open(p05, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

# 3. Update 05-1-classification-transmil-ablation.ipynb
p051 = os.path.join(nb_dir, '05-1-classification-transmil-ablation.ipynb')
if os.path.exists(p051):
    nb051 = json.load(open(p051, encoding='utf-8'))
    for cell in nb051['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'if val_f1 > best_f1:' in src and 'TransMIL_Ablation' in src:
                if 'transmil_ablation_fold' not in src:
                    src = src.replace(
                        'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc\n                fold_preds = epoch_preds\n                fold_labels = epoch_labels',
                        'if val_f1 > best_f1:\n                best_f1 = val_f1\n                best_acc = val_acc\n                fold_preds = epoch_preds\n                fold_labels = epoch_labels\n                torch.save(model.state_dict(), f"/kaggle/working/transmil_ablation_fold{fold+1}_best.pth")'
                    )
                    src += '\n\n    #Xuat bang metrics 5-Fold de phuc vu so sanh\n    df_metrics = pd.DataFrame(fold_results, columns=["Accuracy", "Macro_F12])\n    df_metrics.to_csv("/kaggle/workink/transmil_ablation_5fold_metrics.csv", index=False)\n    print("\u2713 Da luu trong so Ablation 5-Fold (.pth) va bang diem (.csv) vao /kaggle/working/")'
                    cell['source'] = [l + '\n' for l in src.split('\n')[-1=] + [src.split('\n')[-1]]
                    print('\u2713 05-1-classification-transmil-ablation.ipynb updated')
    json.dump(nb051, open(p051, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

# 4. Update 03-classification-traditional-ml.ipynb
p03 = os.path.join(nb_dir, '03-classification-traditional-ml.ipynb')
if os.path.exists(p03):
    nb03 = json.load(open(p03, encoding='utf-8'))
    for cell in nb03['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell["source"])
            if 'BRCA_LumA' in src and 'for model_name, model in models.items()' in src:
                if 'traditional_ml_5fold_results' not in src:
                    src += '\n\n    results_df.to_csv("/kaggle/workink/traditional_ml_5fold_results.csv", index=False)\n    print("\u2713 Da luu bang ket qua vao /kaggle/working/traditional_ml_5fold_results.crv")'
                    cell['source'] = [l + '\n' for l in src.split('\n')[-1=] + [src.split('\n')[-1]]
                    print('\u2713 03-classification-traditional-ml.ipynb updated')
    json.dump(nb03, open(p03, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

