import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- ĐẢM BẢO BẠN ĐÃ CÓ CÁC BIẾN SAU TỪ QUÁ TRÌNH TRAIN ---
# model: Mô hình đã được huấn luyện xong
# X_test, Y_test: Tập dữ liệu test (Dạng Tensor)
# scaler_Y: Bộ biến đổi chuẩn hóa của cột Y (Đã fit ở file preprocessing)
# device: 'cuda' hoặc 'cpu'

# ==========================================
# 1. CHẠY SUY LUẬN TRÊN TẬP TEST (INFERENCE)
# ==========================================
model.eval() # Chuyển model sang chế độ đánh giá (Tắt Dropout)
with torch.no_grad():
    # Đưa X_test vào model để lấy dự đoán
    # Nhớ chuyển vị trí chiều (Batch, Channels, Seq) nếu model yêu cầu
    predictions_tensor = model(X_test.to(device))

# Đưa kết quả từ GPU về lại CPU và chuyển thành mảng Numpy
Y_pred_scaled = predictions_tensor.cpu().numpy()
Y_test_scaled = Y_test.cpu().numpy()

# ==========================================
# 2. GIẢI CHUẨN HÓA (INVERSE TRANSFORM) VỀ RADIAN
# ==========================================
Y_pred_real = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_real = scaler_Y.inverse_transform(Y_test_scaled)

# Tách riêng Roll và Pitch
# Nhắc lại cấu trúc: Cột 0 là Roll, Cột 1 là Pitch
roll_pred, pitch_pred = Y_pred_real[:, 0], Y_pred_real[:, 1]
roll_true, pitch_true = Y_test_real[:, 0], Y_test_real[:, 1]

# Tạo trục thời gian giả lập (100Hz = 0.01s mỗi bước)
time_axis = np.arange(len(roll_true)) * 0.01

# ==========================================
# 3. TÍNH TOÁN ĐIỂM SỐ CHUẨN (METRICS)
# ==========================================
rmse_roll = np.sqrt(mean_squared_error(roll_true, roll_pred))
rmse_pitch = np.sqrt(mean_squared_error(pitch_true, pitch_pred))

print("--- KẾT QUẢ ĐÁNH GIÁ (TEST SET) ---")
print(f"RMSE Roll  : {rmse_roll:.4f} Radian")
print(f"RMSE Pitch : {rmse_pitch:.4f} Radian")

# ==========================================
# 4. VẼ ĐỒ THỊ (PLOTTING)
# ==========================================
# Tạo một khung ảnh lớn chứa 2 biểu đồ xếp chồng lên nhau
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Đồ thị 1: Góc Roll
ax1.plot(time_axis, roll_true, label='Ground Truth (PX4)', color='blue', linewidth=2)
ax1.plot(time_axis, roll_pred, label='AI Prediction (CNN-LSTM)', color='red', linestyle='dashed', linewidth=2)
ax1.set_title(f'Góc Roll: Thực tế vs AI Dự đoán (RMSE: {rmse_roll:.4f})', fontsize=14, fontweight='bold')
ax1.set_ylabel('Radian', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.7)

# Đồ thị 2: Góc Pitch
ax2.plot(time_axis, pitch_true, label='Ground Truth (PX4)', color='green', linewidth=2)
ax2.plot(time_axis, pitch_pred, label='AI Prediction (CNN-LSTM)', color='orange', linestyle='dashed', linewidth=2)
ax2.set_title(f'Góc Pitch: Thực tế vs AI Dự đoán (RMSE: {rmse_pitch:.4f})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Thời gian (Giây)', fontsize=12)
ax2.set_ylabel('Radian', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.7)

# Tinh chỉnh khoảng cách và hiển thị
plt.tight_layout()
plt.show()

# (Tùy chọn) Lưu ảnh lại trên Kaggle
# plt.savefig('/kaggle/working/AI_Benchmark_Result.png', dpi=300)