# 🛍️ Dự đoán cụm khách hàng bằng K-Means

Ứng dụng web đơn giản sử dụng thuật toán **K-Means Clustering** để phân loại khách hàng dựa trên **thu nhập hàng năm** và **điểm chi tiêu**, được xây dựng bằng **Flask**.

---

## 🚀 Tính năng

- Nhập thông tin khách hàng: `Annual Income (k$)` (thu nhập hàng năm) và `Spending Score (1-100)` (điểm chi tiêu)
- Dự đoán cụm khách hàng tương ứng (Cluster 0 → 4)

---

## 🧰 Công nghệ sử dụng

- Python 3
- Flask
- Bootstrap 5
- Pickle

---

## 📁 Nội dung các file

- templates/index.html: Giao diện chính
- kmeans.pkl: Mô hình KMeans đã huấn luyện
- scaler.pkl: Đối tượng chuẩn hóa StandardScaler
- app.py: Flask backend

---

## ✨ Giao diện

<img width="1292" height="595" alt="image" src="https://github.com/user-attachments/assets/51a34c5c-4dc5-4fed-9b7d-a71198984f77" />


---

## ⚙️ Cài đặt & chạy ứng dụng

### 1. Clone dự án

```bash
git clone https://github.com/mi-T-T/PhanCumKhachHang.git
cd PhanCumKhachHang
```

### 2. Chạy ứng dụng

```bash
python app.py
```
