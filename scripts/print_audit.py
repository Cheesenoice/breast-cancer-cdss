import json
import os

with open(r"C:\Users\huynh\Desktop\breast cancer\nb_audit_summary.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

for nb in data:
    print("="*70)
    print(f"FILE: {nb['name']}")
    print(f"MD Cells: {nb['md_count']} | Code Cells: {nb['code_count']}")
    print(f"Headers: {nb['md_headers']}")
    print("--- KEY OUTPUTS ---")
    for out in nb['key_outputs']:
        print(f"[Cell {out['cell_idx']}]:\n{out['text'][:500]}")
        print("-" * 40)
