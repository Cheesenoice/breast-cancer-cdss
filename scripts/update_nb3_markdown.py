import json
import os

path = r'C:\Users\huynh\Desktop\breast cancer\03_Classification_Traditional_ML.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and '### 3. Huấn luyện và Đánh giá (5-Fold Cross Validation)' in ''.join(cell.get('source', [])):
        new_source = [
            "### 3. Huấn luyện và Đánh giá (5-Fold Cross Validation)\n",
            "**Tại sao dùng Stratified 5-Fold?** \n",
            "Kỹ thuật này bẻ tập dữ liệu thành 5 phần (mỗi phần 20%). Mô hình sẽ lần lượt lấy 4 phần để học (Train) và 1 phần để thi (Test). Chữ `Stratified` đảm bảo tỷ lệ số lượng bệnh nhân nhóm LumA, HER2... ở trong tập Test luôn giống hệt với tỷ lệ của tập dữ liệu thực tế.\n",
            "\n",
            "**💡 Giải thích cặn kẽ cách chia (Ví dụ bảo vệ hội đồng):**\n",
            "Giả sử hệ thống của chúng ta có 1000 bệnh nhân, trong đó bị lệch nặng: **800 ca LumA (chiếm 80%)** và chỉ có **50 ca HER2 (chiếm 5%)**.\n",
            "- ❌ **Nếu dùng K-Fold thông thường (bốc ngẫu nhiên):** Rất có thể trong 1 tập Test (200 bệnh nhân), mô hình \"vô tình\" bốc trúng toàn ca dễ, không có ca HER2 nào. Lúc này điểm Accuracy báo cáo rất cao, nhưng thực chất mô hình chưa hề đụng độ ca khó HER2 nào. Đây là lỗi \"ăn may\" chết người trong nghiên cứu y khoa.\n",
            "- ✅ **Nếu dùng Stratified K-Fold:** Thuật toán sẽ ép buộc mỗi tập Test (200 bệnh nhân) phải chia theo đúng tỷ lệ sinh học gốc: Tức là chắc chắn phải rút ra **160 ca LumA (80%)** và **đúng 10 ca HER2 (5%)**. \n",
            "\n",
            "👉 Nhờ cơ chế chia khóa cứng tỷ lệ này, ở bất kỳ vòng thi nào (Fold 1 đến Fold 5), mô hình cũng **bắt buộc phải đối mặt với đầy đủ các loại bệnh nhân**. Điểm số cuối cùng mà chúng ta báo cáo ra là điểm thực lực 100%, không thể lách luật hay ăn may được!"
        ]
        cell['source'] = new_source
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated Notebook 3 markdown successfully.")
