import json
import os

path = r'C:\Users\huynh\.gemini\antigravity\brain\884de7a7-c6e3-4db1-a937-fbde5508cf2a\implementation_plan.md'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Thay thế mô tả Notebook 4 cũ
old_nb4 = """#### [NEW] `04_Classification_CNN_FineTuning.ipynb` (Thành viên D)
- Train trực tiếp một mô hình CNN (Fine-tuning MobileNetV3 hoặc ResNet) trên ảnh Patch với kỹ thuật Majority Voting ở khâu cuối. Mục tiêu là tạo ra một Baseline Deep Learning so sánh."""

new_nb4 = """#### [NEW] `04_Classification_Naive_DeepMIL.ipynb` (Thành viên D) - *Đã cập nhật theo yêu cầu*
- **Mô hình Deep Learning Baseline (Naive MIL):** Thay vì dùng ML truyền thống (NB3), ta xây dựng một mạng Neural Network đa tầng (MLP/Fully Connected).
- **Cơ chế:** Lấy các file tensor `.pt`, cộng dồn và ép phẳng bằng `Mean-Pooling` hoặc `Max-Pooling` để ra 1 siêu vector (2048 chiều). Sau đó đưa qua các lớp Linear (Dropout, ReLU) để phân loại 4 nhãn.
- **Mục đích khoa học:** Chứng minh lập luận rằng sự yếu kém không nằm ở "sức mạnh thuật toán", mà nằm ở chỗ kỹ thuật "Cộng trung bình thô thiển" đã cào bằng và xóa sổ các đặc điểm tế bào ung thư quý giá. Sự sụp đổ của mạng Neural Network này sẽ là bàn đạp lý tưởng để giới thiệu siêu phẩm TransMIL ở Notebook 5."""

if old_nb4 in content:
    content = content.replace(old_nb4, new_nb4)
else:
    # Fallback in case string exact match fails, use regex or replace Phase 2 entirely.
    import re
    content = re.sub(
        r'#### \[NEW\] `04_Classification_CNN_FineTuning.ipynb`.*?(?=#### \[NEW\] `05_)', 
        new_nb4 + '\n\n', 
        content, 
        flags=re.DOTALL
    )

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated Implementation Plan with Naive Deep MIL successfully.")
