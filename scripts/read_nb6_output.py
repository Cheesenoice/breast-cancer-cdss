import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\nb-w-output\06-0-multimodal-eda-and-genomics-processing.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        outputs = cell.get('outputs', [])
        print(f"--- [Cell {i} Outputs] ---")
        for out in outputs:
            if out['output_type'] == 'stream':
                print("".join(out.get('text', [])))
            elif out['output_type'] in ['display_data', 'execute_result']:
                data = out.get('data', {})
                if 'text/plain' in data:
                    print("".join(data['text/plain']))
                if 'image/png' in data:
                    print("[Image/PNG generated successfully]")
