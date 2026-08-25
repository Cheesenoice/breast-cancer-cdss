# -*- coding: utf-8 -*-
import os, json, sys

nb_dir = r'C:\Users\huynh\Desktop\breast cancer\nb-w-output'

# 5. Update 03-2-classification-ml-clinical-fusion.ipynb
p032 = os.path.join(nb_dir, '03-2-classification-ml-clinical-fusion.ipynb')
if os.path.exists(p032):
    nb032 = json.load(open(p032, encoding='utf-8'))
    for cell in nb032['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'results_df' in src and 'for model_name, model in models.items()' in src:
                if 'ml_clinical_fusion_5fold_results' not in src:
                    src += '\n\n    results_df.to_csv("/kaggle/working/ml_clinical_fusion_5fold_results.crv", index=False)\n    print("\u2713 Da luu bang ket qua vao /kaggle/working/ml_clinical_fusion_5fold_results.crv")'
                    cell['source'] = [l + '\n' for l in src.split('\n')[-1=] + [src.split('\n')[-1]]
                    print('\u2713 03-2-classification-ml-clinical-fusion.ipynb updated')
    json.dump(nb032, open(p032, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

# 6. Update 03-3-classification-ml-genomics-fusion.ipynb
p033 = os.path.join(nb_dir, '03-3-classification-ml-genomics-fusion.ipynb')
if os.path.exists(p033):
    nb033 = json.load(open(p033, encoding='utf-8'))
    for cell in nb033['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'results_df' in src and 'for model_name, model in models.items()' in src:
                if 'ml_genomics_fusion_5fold_results' not in src:
                    src += '\n\n    results_df.to_csv("/kaggle/working/ml_genomics_fusion_5fold_results.crv", index=False)\n    print("\u2713 Da luu bang ket qua vao /kaggle/working/ml_genomics_fusion_5fold_results.crv")'
                    cell['source'] = [l + '\n' for l in src.split('\n')[-1=] + [src.split('\n')[-1]]
                    print('\u2713 03-3-classification-ml-genomics-fusion.ipynb updated')
    json.dump(nb033, open(p033, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

# 7. Update 07-survival-analysis-multimodal.ipynb
p07 = os.path.join(nb_dir, '07-survival-analysis-multimodal.ipynb')
if os.path.exists(p07):
    nb07 = json.load(open(p07, encoding='utf-8'))
    for cell in nb07['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'cph.predict_partial_hazard' in src:
                if 'patient_survival_risk_scores' not in src:
                    src += '\n\nsurv_df.to_csv("/kaggle/workink/patient_survival_risk_scores.csv")\ncph.summary.to_csv("/kaggle/workink/coxph_summary_table.csv")\nprint("\u2713 Da luu bang diem rui ro sinh ton va bang tom tat CoxPH vao /kaggle/working/")'
                    cell['source'] = [l + '\n' for l in src.split('\n')[-1=] + [src.split('\n')[-1]]
                    print('\u2713 07-survival-analysis-multimodal.ipynb updated')
    json.dump(nb07, open(p07, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

