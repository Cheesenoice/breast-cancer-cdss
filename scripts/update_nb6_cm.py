import json
import os

def update_notebook_with_cm(path, title):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'if len(valid_pids) > 0:' in source and 'skf = StratifiedKFold' in source:
                new_source = []
                lines = source.split('\n')
                for line in lines:
                    if 'fold_results = []' in line:
                        new_source.append(line)
                        new_source.append("    best_global_preds = []")
                        new_source.append("    best_global_labels = []")
                    elif 'best_f1, best_acc = 0.0, 0.0' in line:
                        new_source.append(line)
                        new_source.append("        best_fold_preds = []")
                        new_source.append("        best_fold_labels = []")
                    elif 'best_f1, best_acc = val_f1, accuracy_score(tgts, preds)' in line:
                        new_source.append(line)
                        new_source.append("                best_fold_preds = preds")
                        new_source.append("                best_fold_labels = tgts")
                    elif 'fold_results.append((best_acc, best_f1))' in line:
                        new_source.append(line)
                        new_source.append("        best_global_preds.extend(best_fold_preds)")
                        new_source.append("        best_global_labels.extend(best_fold_labels)")
                    else:
                        new_source.append(line)
                cell['source'] = '\n'.join(new_source)

    # Thêm cell vẽ Confusion Matrix
    cm_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
            "if len(valid_pids) > 0:\n",
            "    cm = confusion_matrix(best_global_labels, best_global_preds)\n",
            "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['LumA', 'LumB', 'Basal', 'HER2'])\n",
            "    \n",
            "    fig, ax = plt.subplots(figsize=(8, 6))\n",
            "    disp.plot(cmap='Blues', ax=ax, values_format='d')\n",
            f"    plt.title('Bản đồ Confusion Matrix - {title}', fontsize=14, fontweight='bold')\n",
            "    plt.show()"
        ]
    }
    
    # Kiểm tra xem đã có cell confusion matrix chưa
    has_cm = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'ConfusionMatrixDisplay' in ''.join(cell.get('source', [])):
            has_cm = True
            break
            
    if not has_cm:
        nb['cells'].append(cm_cell)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

update_notebook_with_cm(r'C:\Users\huynh\Desktop\breast cancer\06.1_Multimodal_WSI_Clinical.ipynb', 'Multimodal (WSI + Lâm sàng)')
update_notebook_with_cm(r'C:\Users\huynh\Desktop\breast cancer\06.2_Multimodal_WSI_Genomics.ipynb', 'Multimodal God Mode (WSI + RNA-Seq)')
print("Thêm Confusion Matrix thành công.")
