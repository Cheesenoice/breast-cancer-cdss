import os
import json

nb_dir = r"C:\Users\huynh\Desktop\breast cancer\nb-w-output"
notebooks = sorted([f for f in os.listdir(nb_dir) if f.endswith('.ipynb')])

summary = []

for nb_name in notebooks:
    path = os.path.join(nb_dir, nb_name)
    with open(path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            summary.append(f"Error loading {nb_name}: {e}")
            continue

    cells = nb.get('cells', [])
    md_cells = []
    code_outputs = []
    
    for i, cell in enumerate(cells):
        ctype = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        
        if ctype == 'markdown':
            # Collect first line or headers
            headers = [line for line in source.split('\n') if line.strip().startswith('#')]
            md_cells.append({
                'index': i,
                'headers': headers,
                'preview': source[:200].replace('\n', ' ')
            })
        elif ctype == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                out_type = out.get('output_type')
                if out_type == 'stream':
                    text = "".join(out.get('text', []))
                    # check if it has key metrics
                    if any(k in text for k in ['Accuracy', 'F1', 'Macro-F1', 'TỔNG KẾT', 'Fold', 'C-Index', 'p-value', 'Kỷ lục', 'BẢNG XẾP HẠNG']):
                        code_outputs.append({
                            'cell_idx': i,
                            'text': text.strip()
                        })
                elif out_type in ['execute_result', 'display_data']:
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        txt = "".join(data['text/plain'])
                        if any(k in txt for k in ['Accuracy', 'F1', 'Macro-F1', 'TỔNG KẾT', 'Fold', 'C-Index', 'p-value']):
                            code_outputs.append({
                                'cell_idx': i,
                                'text': txt.strip()
                            })

    summary.append({
        'name': nb_name,
        'num_cells': len(cells),
        'md_count': len([c for c in cells if c.get('cell_type') == 'markdown']),
        'code_count': len([c for c in cells if c.get('cell_type') == 'code']),
        'md_headers': [h for m in md_cells for h in m['headers']],
        'key_outputs': code_outputs
    })

# Output summary to a text file for inspection
with open(r"C:\Users\huynh\Desktop\breast cancer\nb_audit_summary.json", 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Audited {len(notebooks)} notebooks successfully.")
