import os
import json

nb_dir = r"C:\Users\huynh\Desktop\breast cancer\nb-w-output"

def inspect_notebook_md(name):
    path = os.path.join(nb_dir, name)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("="*80)
    print(f"NOTEBOOK: {name}")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            print(f"--- [MD Cell {i}] ---")
            print("".join(cell.get('source', [])))

inspect_notebook_md("03-3-classification-ml-genomics-fusion.ipynb")
inspect_notebook_md("06-2-multimodal-wsi-genomics.ipynb")
inspect_notebook_md("08-explainable-ai-multimodal.ipynb")
