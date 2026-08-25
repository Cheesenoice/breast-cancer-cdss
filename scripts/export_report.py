import json
import os

with open(r"C:\Users\huynh\Desktop\breast cancer\nb_audit_summary.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(r"C:\Users\huynh\Desktop\breast cancer\full_audit_report.txt", 'w', encoding='utf-8') as out_f:
    for nb in data:
        out_f.write("="*80 + "\n")
        out_f.write(f"FILE: {nb['name']}\n")
        out_f.write(f"MD Cells: {nb['md_count']} | Code Cells: {nb['code_count']}\n")
        out_f.write(f"Headers:\n")
        for h in nb['md_headers']:
            out_f.write(f"  - {h}\n")
        out_f.write("\n--- KEY OUTPUTS ---\n")
        for out in nb['key_outputs']:
            out_f.write(f"[Cell {out['cell_idx']}]:\n{out['text']}\n")
            out_f.write("-" * 50 + "\n")
        out_f.write("\n")

print("Generated full_audit_report.txt")
