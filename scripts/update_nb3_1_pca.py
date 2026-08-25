import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\03.1_Classification_Traditional_ML_Advanced.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and '### 2. Tối Ưu Hóa Đặc Trưng bằng PCA' in ''.join(cell.get('source', [])):
        new_source = [
            "### 2. Tối Ưu Hóa Đặc Trưng bằng PCA (Dimensionality Reduction)\n",
            "\n",
            "**Tại sao bắt buộc phải dùng PCA (Principal Component Analysis)?**\n",
            "Sau bước nhào nặn đặc trưng (Mean + Max + Std) ở trên, chúng ta đã tạo ra một ma trận khổng lồ `[Số bệnh nhân, 6144 chiều]`. Tuy nhiên, đưa nguyên 6144 chiều này vào huấn luyện sẽ gây ra 2 rủi ro chết người:\n",
            "1. **Lời nguyền chiều dữ liệu (Curse of Dimensionality):** Khi số chiều (6144) lớn hơn rất nhiều so với số điểm dữ liệu (882 bệnh nhân), các mô hình ML sẽ bị \"lạc lối\", dẫn đến hiện tượng học vẹt (Overfitting) nghiêm trọng. Chúng sẽ học thuộc lòng các nhiễu thay vì học quy luật.\n",
            "2. **Nhiễu thông tin khổng lồ (Noise):** Rất nhiều chiều trong 6144 biến số kia là rác (ví dụ: các vùng ảnh chỉ chứa background trắng, bong bóng khí, hoặc mô mỡ vô hại). Giữ chúng lại chỉ làm nhiễu mô hình.\n",
            "\n",
            "**💡 Sự tương đồng lý luận với bài báo của Thầy:**\n",
            "Trong bài báo nghiên cứu lá sầu riêng, Thầy đã giải quyết vấn đề nhiễu đặc trưng này bằng cách dùng **Thuật toán Bầy đàn (Particle Swarm Optimization - PSO)** để tỉa bớt (Pruning) các đặc trưng thừa. Kế thừa triết lý \"Feature Optimization\" đó, trong đồ án y tế này, chúng ta sử dụng **PCA** - một phương pháp biến đổi đại số tuyến tính kinh điển - để thực thi nhiệm vụ thanh lọc.\n",
            "\n",
            "**PCA hoạt động như thế nào?**\n",
            "Thuật toán PCA không \"xóa\" ngẫu nhiên các cột. Nó tính toán Ma trận hiệp phương sai (Covariance Matrix) và xoay trục không gian dữ liệu để tìm ra các hướng có độ biến thiên (Variance) lớn nhất.\n",
            "- Nó nén ép 6144 chiều thô kệch xuống còn **128 chiều (Principal Components)**.\n",
            "- 👉 **Kết quả:** Dù dung lượng bị nén đi 48 lần, nhưng 128 chiều mới này lại là những hạt nhân tinh túy nhất, giữ lại được phần lớn lượng thông tin sinh học cốt lõi (Explained Variance). Nhờ PCA, các thuật toán như SVM hay XGBoost sẽ chạy cực nhanh và có độ tập trung sắc bén vào các dị dạng ung thư mà không bị xao nhãng bởi rác!"
        ]
        cell['source'] = new_source
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Notebook 3.1 PCA markdown successfully.")
