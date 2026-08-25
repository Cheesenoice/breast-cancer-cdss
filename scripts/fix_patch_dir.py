import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\08_Explainable_AI_Multimodal.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'PATCH_DIR =' in ''.join(cell.get('source', [])):
        new_source = """from PIL import Image
import glob

# Đã cập nhật đúng đường dẫn Patches gốc của bạn
PATCH_DIR = '/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/WSIs'
patient_patch_dir = os.path.join(PATCH_DIR, demo_pid)

if os.path.exists(patient_patch_dir):
    # Lấy tất cả các file .jpg và loại bỏ thư mục rác checkpoint
    all_patches = [f for f in os.listdir(patient_patch_dir) if f.endswith('.jpg') and 'checkpoint' not in f]
    
    # Sắp xếp ảnh: Ưu tiên sắp xếp theo số nguyên (0.jpg -> 1.jpg -> 2.jpg thay vì 0.jpg -> 10.jpg -> 2.jpg)
    try:
        all_patches = sorted(all_patches, key=lambda x: int(x.split('.')[0]))
    except:
        all_patches = sorted(all_patches)
    
    # Lấy ra Top 5 mảnh quan trọng nhất từ mảng patch_importance
    top_5_indices = np.argsort(patch_importance)[::-1][:5]
    
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(top_5_indices):
        if idx < len(all_patches):
            img_path = os.path.join(patient_patch_dir, all_patches[idx])
            try:
                img = Image.open(img_path)
                plt.subplot(1, 5, i + 1)
                plt.imshow(img)
                plt.title(f"Rank #{i+1}\\nScore: {patch_importance[idx]:.4f}", fontsize=12, fontweight='bold', color='red')
                plt.axis('off')
            except Exception as e:
                pass
    plt.suptitle(f"Top 5 Mảnh Tế Bào Ác Tính Nhất - Bệnh nhân {demo_pid}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print(f"⚠️ Không tìm thấy thư mục ảnh gốc: {patient_patch_dir}")
"""
        cell['source'] = [new_source]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Đã cập nhật PATCH_DIR và thuật toán lọc ảnh thành công.")
