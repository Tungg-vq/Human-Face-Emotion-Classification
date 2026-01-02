# 🎭 Web App Nhận diện Cảm xúc Khuôn mặt

Ứng dụng web sử dụng Deep Learning để nhận diện cảm xúc từ khuôn mặt trong ảnh.

## 🌟 Tính năng

- ✅ Tự động phát hiện khuôn mặt trong ảnh
- ✅ Nhận diện 5 cảm xúc: Tức giận, Sợ hãi, Vui vẻ, Buồn bã, Ngạc nhiên
- ✅ Hiển thị độ tin cậy của mỗi dự đoán
- ✅ Giao diện thân thiện, dễ sử dụng
- ✅ Hỗ trợ nhiều khuôn mặt trong một ảnh

## 🚀 Cài đặt

### Bước 1: Clone repository (nếu chưa có)
```bash
git clone <repository-url>
cd Human-Face-Emotion-Classification
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements_webapp.txt
```

Hoặc nếu bạn muốn dùng file requirements.txt gốc:
```bash
pip install -r requirements.txt
pip install streamlit
```

### Bước 3: Đảm bảo có file model
Kiểm tra xem file `model/best_model_acc_final.pth` đã tồn tại hay chưa.

## 🎮 Chạy ứng dụng

Chạy lệnh sau trong terminal:

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: `http://localhost:8501`

## 📖 Hướng dẫn sử dụng

1. **Tải ảnh lên**: Click vào nút "Browse files" để chọn ảnh từ máy tính
2. **Xem kết quả**: Ứng dụng sẽ tự động:
   - Phát hiện tất cả khuôn mặt trong ảnh
   - Vẽ khung màu quanh mỗi khuôn mặt
   - Hiển thị cảm xúc và độ tin cậy
3. **Chi tiết**: Mở rộng từng khuôn mặt để xem thông tin chi tiết

## 🎨 Các cảm xúc được nhận diện

| Cảm xúc | Màu sắc | Icon |
|---------|---------|------|
| Tức giận (Angry) | 🔴 Đỏ | 😠 |
| Sợ hãi (Fear) | 🟣 Tím | 😨 |
| Vui vẻ (Happy) | 🟢 Xanh lá | 😊 |
| Buồn bã (Sad) | 🔵 Xanh dương | 😢 |
| Ngạc nhiên (Surprise) | 🟠 Cam | 😲 |

## 🛠️ Công nghệ sử dụng

- **PyTorch**: Framework Deep Learning
- **Streamlit**: Framework Web App
- **OpenCV**: Xử lý ảnh và phát hiện khuôn mặt
- **CNN**: Mạng Neural tích chập tùy chỉnh

## 📁 Cấu trúc thư mục

```
Human-Face-Emotion-Classification/
├── app.py                          # File chính của web app
├── requirements_webapp.txt         # Dependencies cho web app
├── README_WEBAPP.md               # File này
├── model/
│   ├── best_model_acc_final.pth   # Model đã train
│   └── training.ipynb             # Notebook training
└── data/                          # Dữ liệu training (nếu có)
```

## ⚙️ Tùy chỉnh

### Thay đổi cổng mặc định
```bash
streamlit run app.py --server.port 8080
```

### Chạy ở chế độ production
```bash
streamlit run app.py --server.headless true
```

### Cấu hình nâng cao
Tạo file `.streamlit/config.toml`:
```toml
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🐛 Xử lý sự cố

### Lỗi: "Module not found"
```bash
pip install --upgrade -r requirements_webapp.txt
```

### Lỗi: "Model not found"
Đảm bảo file model nằm đúng vị trí: `model/best_model_acc_final.pth`

### Lỗi: "No faces detected"
- Đảm bảo ảnh có chứa khuôn mặt rõ ràng
- Khuôn mặt nên nhìn thẳng hoặc nghiêng nhẹ
- Ánh sáng đủ để nhận diện

## 📝 Ghi chú

- Model hoạt động tốt nhất với ảnh khuôn mặt rõ nét
- Hỗ trợ GPU nếu có (tự động phát hiện CUDA)
- Có thể xử lý nhiều khuôn mặt trong một ảnh

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo Pull Request hoặc mở Issue.

## 📄 License

MIT License

---

**Phát triển với ❤️ bằng PyTorch và Streamlit**
