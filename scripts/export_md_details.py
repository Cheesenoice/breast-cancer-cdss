import os
import json

nb_dir = r"C:\Users\huynh\Desktop\breast cancer\nb-w-output"

with open(r"C:\Users\huynh\Desktop\breast cancer\inspected_md_details.txt", 'w', encoding='utf-8') as out:
    for name in ["03-2-classification-ml-clinical-fusion.ipynb", "03-3-classification-ml-genomics-fusion.ipynb", "06-1-multimodal-wsi-clinical.ipynb", "06-2-multimodal-wsi-genomics.ipynb", "07-survival-analysis-multimodal.ipynb", "08-explainable-ai-multimodal.ipynb"]:
        path = os.path.join(nb_dir, name)
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        out.write("="*80 + "\n")
        out.write(f"NOTEBOOK: {name}\n")
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                out.write(f"--- [MD Cell {i}] ---\n")
                out.write("".join(cell.get('source', [])) + "\n\n")

print("Saved inspected_md_details.txt")
