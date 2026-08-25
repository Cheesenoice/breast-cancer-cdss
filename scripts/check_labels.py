import csv
from collections import Counter

csv_path = r"C:\Users\huynh\Desktop\gen\data\tcga_brca\tcga_brca_master_matched_cohort.csv"

try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        
        if len(rows) > 0 and len(rows[0].keys()) < 5:
            f.seek(0)
            reader = csv.DictReader(f, delimiter=',')
            rows = list(reader)
            
    print(f"Total rows: {len(rows)}")
    
    if rows and 'pam50_subtype' in rows[0]:
        print("\n--- pam50_subtype ---")
        counts = Counter(r['pam50_subtype'] for r in rows)
        for k, v in counts.items():
            print(f"{k}: {v}")
    elif rows and 'SUBTYPE' in rows[0]:
        print("\n--- SUBTYPE ---")
        counts = Counter(r['SUBTYPE'] for r in rows)
        for k, v in counts.items():
            print(f"{k}: {v}")
    else:
        print("Columns found:")
        if rows:
            print(list(rows[0].keys()))

except Exception as e:
    print(f"Error: {e}")
