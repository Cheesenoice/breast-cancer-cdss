import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\03_Classification_Traditional_ML.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and '### 2. Thiết lập các Mô hình Machine Learning Truyền thống' in ''.join(cell.get('source', [])):
        new_source = [
            "### 2. Thiết lập các Mô hình Machine Learning Truyền thống\n",
            "Để phục vụ mục đích Báo cáo & Phản biện, dưới đây là lý luận chuyên sâu về các thiết lập (Hyperparameters) cốt lõi của hệ thống:\n",
            "\n",
            "#### A. Cơ chế Cân bằng Nhãn (Class Weighting)\n",
            "Thay vì dùng kỹ thuật sinh ảnh giả (SMOTE) làm sai lệch cấu trúc sinh học, chúng ta sử dụng tham số `class_weight='balanced'`. Thuật toán sẽ tính trọng số phạt (Loss Penalty) cho từng nhóm dựa trên độ hiếm của nó theo công thức:\n",
            "> **Trọng số = Tổng số bệnh nhân / (Số lượng nhãn × Số bệnh nhân của nhãn đó)**\n",
            "\n",
            "**Ví dụ thực tế (Cho Báo cáo/Thuyết trình):**\n",
            "Giả sử hệ thống có 1000 bệnh nhân, trong đó có 800 ca LumA (chiếm 80%, rất đông) và 50 ca HER2 (chiếm 5%, rất hiếm). Máy tính sẽ gán trọng số phạt như sau:\n",
            "- Trọng số phạt cho LumA: `1000 / (4 × 800) = 0.3125`\n",
            "- Trọng số phạt cho HER2: `1000 / (4 × 50) = 5.0`\n",
            "\n",
            "👉 **Ý nghĩa:** Nếu AI đoán sai 1 ca LumA, nó bị trừ rất nhẹ (0.3125 điểm). Nhưng nếu đoán sai 1 ca HER2, nó bị phạt nặng gấp 16 lần (5.0 điểm). Điều này ép mô hình từ bỏ thói quen đoán bừa nhóm đông (LumA) và bắt buộc phải chú ý quan sát mảnh tế bào của nhóm hiếm (HER2).\n",
            "\n",
            "#### B. Giải thích Cấu hình Thuật toán (Hyperparameters)\n",
            "1. **Random Forest (`n_estimators=100`):** Xây dựng 100 cây quyết định (Decision Trees) khác nhau và cho chúng biểu quyết. Số 100 là điểm cân bằng lý tưởng (Sweet-spot) để chống Overfitting trên không gian dữ liệu khổng lồ (2048 chiều) mà không làm tràn RAM.\n",
            "2. **SVM (`kernel='rbf'`):** RBF (Radial Basis Function) là hạt nhân phi tuyến tính cực mạnh. Vector 2048 chiều từ ảnh hiếm khi có ranh giới phân chia thẳng. RBF giúp chiếu dữ liệu lên không gian đa chiều cao hơn để tìm ra nếp gấp phân tách các nhóm ung thư một cách tinh tế. Đây cũng là cấu hình chuẩn được dùng trong bài báo của Thầy.\n",
            "3. **Logistic Regression (`max_iter=1000`):** Thuật toán tuyến tính kinh điển. Vì không gian lên tới 2048 chiều, thuật toán cần nhiều vòng lặp hơn để hội tụ, do đó ta ép `max_iter` lên 1000 thay vì 100 như mặc định để tránh lỗi chưa hội tụ (ConvergenceWarning).\n",
            "4. **XGBoost (`eval_metric='mlogloss'`):** Multi-class Log Loss là độ đo tối ưu nhất để phạt XGBoost khi nó tự tin đưa ra xác suất sai. (Lưu ý: Do XGBoost phiên bản mới xử lý class_weight qua sample_weight, chúng ta tính toán riêng mảng `weights` cho nó).\n",
            "5. **Voting Ensemble (`voting='soft'`):** Bầu chọn mềm (Soft Voting) mạnh hơn Bầu chọn cứng (Hard Voting). Thay vì mỗi mô hình giơ 1 phiếu bầu (Hard), Soft Voting thu thập tỷ lệ % xác suất của 4 mô hình cộng lại (Ví dụ: RF đoán 40% LumA, SVM đoán 55% LumA -> Trung bình 47.5%). Điều này giúp triệt tiêu các sai số cá nhân và tạo ra dự đoán mượt mà, chính xác nhất."
        ]
        cell['source'] = new_source
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Notebook 3 parameters markdown successfully.")
