import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG (4 BIẾN, WINDOW = 20)
# ==========================================
class UAV_Torque_Net(nn.Module):
    def __init__(self, seq_length=20):
        super(UAV_Torque_Net, self).__init__()
        self.cnn_block = nn.Sequential(
            # CHÚ Ý: in_channels = 4
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) 
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.4)
        self.fc_block = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 2))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn_block(x) 
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :] 
        return self.fc_block(last_time_step_out)

# ==========================================
# 2. ĐỌC VÀ TIỀN XỬ LÝ TẬP DỮ LIỆU MỚI TOANH
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

# THAY ĐỔI TÊN FILE Ở ĐÂY NẾU CẦN
csv_file_path = 'AI_Training_Data.csv' 
try:
    df = pd.read_csv(csv_file_path)
    print(f"Đã tải thành công tập dữ liệu test mới: {len(df)} dòng.")
except FileNotFoundError:
    # Nếu đang để ở thư mục working thì dùng dòng dưới
    df = pd.read_csv('/kaggle/working/AI_Training_Data.csv')
    print(f"Đã tải thành công tập dữ liệu test mới: {len(df)} dòng.")

# Tính Momen
df['tau_roll']  = df['m2']**2 + df['m3']**2 - df['m1']**2 - df['m4']**2
df['tau_pitch'] = df['m2']**2 + df['m4']**2 - df['m1']**2 - df['m3']**2

cols_X = ['tau_roll', 'tau_pitch', 'roll', 'pitch'] 
cols_Y = ['roll', 'pitch']

# Lọc nhiễu
FS, CUTOFF = 100.0, 35.0 # Dùng 35.0 để bắt được các đỉnh gắt
df[['tau_roll', 'tau_pitch']] = butter_lowpass_filter(df[['tau_roll', 'tau_pitch']].values, CUTOFF, FS)

# Chuẩn hóa Z-Score (Lưu ý: Thực tế nên load lại scaler cũ, nhưng test nhanh ta fit mới)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
normalized_X = scaler_X.fit_transform(df[cols_X].values)
normalized_Y = scaler_Y.fit_transform(df[cols_Y].values)

# Cắt cửa sổ (Sliding Window)
WINDOW_SIZE = 20
X_test_full, Y_test_full = [], []
for i in range(len(normalized_X) - WINDOW_SIZE):
    X_test_full.append(normalized_X[i : i + WINDOW_SIZE, :])
    Y_test_full.append(normalized_Y[i + WINDOW_SIZE, :])

X_test = np.array(X_test_full)
Y_test = np.array(Y_test_full)

# ==========================================
# 3. NẠP MÔ HÌNH VÀ CHẠY GIẢ LẬP MPO (VIRTUAL IMU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UAV_Torque_Net(seq_length=20).to(device)

# Load cục tạ siêu xịn của bạn
model.load_state_dict(torch.load('/kaggle/working/uav_torque_best.pth'))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
predictions_mpo = []

print("Bắt đầu chạy giả lập MPO trên toàn bộ tập dữ liệu mới...")

with torch.no_grad():
    # Mồi lửa bằng cửa sổ 0.2s đầu tiên
    current_window = X_test_tensor[0].unsqueeze(0).clone()
    
    for i in range(len(X_test_tensor)):
        # 1. AI tự dự đoán Góc (Dạng Scaled)
        pred_Y = model(current_window) 
        predictions_mpo.append(pred_Y.cpu().numpy()[0])
        
        # 2. Cuốn chiếu cửa sổ cho nhịp tiếp theo
        if i < len(X_test_tensor) - 1:
            next_window = X_test_tensor[i+1].unsqueeze(0).clone().to(device)
            
            # --- CẬP NHẬT GÓC (CỰC KỲ ĐƠN GIẢN CHO BẢN 4 BIẾN) ---
            # Vì ta không cần tính đạo hàm, nên có thể nhét thẳng giá trị Scaled dự đoán 
            # vào vị trí [cột 2 là Roll, cột 3 là Pitch] của timestep cuối cùng
            next_window[0, -1, 2] = pred_Y[0, 0] 
            next_window[0, -1, 3] = pred_Y[0, 1] 
            
            current_window = next_window

# ==========================================
# 4. GIẢI CHUẨN HÓA VÀ VẼ ĐỒ THỊ
# ==========================================
Y_pred_scaled = np.array(predictions_mpo)
Y_pred_real = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_real = scaler_Y.inverse_transform(Y_test)

roll_pred, pitch_pred = Y_pred_real[:, 0], Y_pred_real[:, 1]
roll_true, pitch_true = Y_test_real[:, 0], Y_test_real[:, 1]

rmse_roll = np.sqrt(mean_squared_error(roll_true, roll_pred))
rmse_pitch = np.sqrt(mean_squared_error(pitch_true, pitch_pred))

print(f" KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP DỮ LIỆU MỚI:")
print(f"   - RMSE Roll  : {rmse_roll:.4f} Radian")
print(f"   - RMSE Pitch : {rmse_pitch:.4f} Radian")

# --- VẼ ĐỒ THỊ ---
time_axis = np.arange(len(roll_true)) * 0.01 # 100Hz = 0.01s
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(time_axis, roll_true, label='Ground Truth (Dữ liệu mới)', color='blue', linewidth=2, alpha=0.7)
ax1.plot(time_axis, roll_pred, label='AI Prediction (MPO 4 Biến)', color='red', linestyle='dashed', linewidth=2)
ax1.set_title(f'Góc Roll: Test trên Dữ liệu Unseen (RMSE: {rmse_roll:.4f})', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_ylabel('Radian', fontsize=12)

ax2.plot(time_axis, pitch_true, label='Ground Truth (Dữ liệu mới)', color='green', linewidth=2, alpha=0.7)
ax2.plot(time_axis, pitch_pred, label='AI Prediction (MPO 4 Biến)', color='orange', linestyle='dashed', linewidth=2)
ax2.set_title(f'Góc Pitch: Test trên Dữ liệu Unseen (RMSE: {rmse_pitch:.4f})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.set_xlabel('Thời gian (Giây)', fontsize=12)
ax2.set_ylabel('Radian', fontsize=12)

plt.tight_layout()
plt.show()