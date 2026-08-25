import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\nb-w-output\06-0-multimodal-eda-and-genomics-processing.ipynb'

with open(r'C:\Users\huynh\Desktop\breast cancer\nb6_outputs.txt', 'w', encoding='utf-8') as out_f:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            out_f.write(f"\n--- [Cell {i} Outputs] ---\n")
            for out in outputs:
                if out['output_type'] == 'stream':
                    out_f.write("".join(out.get('text', [])))
                elif out['output_type'] in ['display_data', 'execute_result']:
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        out_f.write("".join(data['text/plain']) + "\n")
                    if 'image/png' in data:
                        out_f.write("[Image/PNG generated successfully]\n")
                        
print("Saved outputs to nb6_outputs.txt")
