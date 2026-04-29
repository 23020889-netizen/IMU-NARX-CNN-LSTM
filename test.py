import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from CNN_LSTM import UAV_Torque_Net
from preprocessing import X_test, Y_test, scaler_Y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UAV_Torque_Net(seq_length=20).to(device)
model.load_state_dict(torch.load('/kaggle/working/uav_torque_best.pth'))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
predictions_mpo = []

print("Bắt đầu chạy giả lập MPO (Virtual IMU - Mất hoàn toàn tín hiệu thật)...")

with torch.no_grad():
    # Mồi lửa bằng cửa sổ 0.5s đầu tiên của tập Test
    current_window = X_test_tensor[0].unsqueeze(0).clone()
    
    for i in range(len(X_test_tensor)):
        # 1. AI đoán Góc tương lai
        pred_Y = model(current_window)
        predictions_mpo.append(pred_Y.cpu().numpy()[0])
        
        # 2. Xây dựng cửa sổ cho bước tiếp theo
        if i < len(X_test_tensor) - 1:
            next_window = X_test_tensor[i+1].unsqueeze(0).clone()
            
            # --- CƠ CHẾ MPO: BỊT MẮT CẢM BIẾN ---
            # Chỉ mục cột 2 là Roll, Cột 3 là Pitch. 
            # Đè kết quả vừa dự đoán (pred_Y) vào timestep cuối cùng (-1) của next_window
            next_window[0, -1, 2] = pred_Y[0, 0] # Roll
            next_window[0, -1, 3] = pred_Y[0, 1] # Pitch
            
            current_window = next_window

Y_pred_scaled = np.array(predictions_mpo)

# GIẢI CHUẨN HÓA
Y_pred_real = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_real = scaler_Y.inverse_transform(Y_test)

roll_pred, pitch_pred = Y_pred_real[:, 0], Y_pred_real[:, 1]
roll_true, pitch_true = Y_test_real[:, 0], Y_test_real[:, 1]

rmse_roll = np.sqrt(mean_squared_error(roll_true, roll_pred))
rmse_pitch = np.sqrt(mean_squared_error(pitch_true, pitch_pred))

print(f"MPO RMSE Roll  : {rmse_roll:.4f} Radian")
print(f"MPO RMSE Pitch : {rmse_pitch:.4f} Radian")

# VẼ ĐỒ THỊ
time_axis = np.arange(len(roll_true)) * 0.01
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(time_axis, roll_true, label='Ground Truth', color='blue', linewidth=2, alpha=0.7)
ax1.plot(time_axis, roll_pred, label='AI Prediction (MPO)', color='red', linestyle='dashed', linewidth=2)
ax1.set_title(f'Góc Roll: MPO (RMSE: {rmse_roll:.4f})')
ax1.legend()

ax2.plot(time_axis, pitch_true, label='Ground Truth', color='green', linewidth=2, alpha=0.7)
ax2.plot(time_axis, pitch_pred, label='AI Prediction (MPO)', color='orange', linestyle='dashed', linewidth=2)
ax2.set_title(f'Góc Pitch: MPO (RMSE: {rmse_pitch:.4f})')
ax2.legend()

plt.tight_layout()
plt.show()