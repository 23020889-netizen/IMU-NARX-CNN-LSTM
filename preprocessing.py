import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# ==========================================
# BƯỚC 1: HÀM LỌC NHIỄU (LOW-PASS FILTER)
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    # fs: sampling frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

# ==========================================
# BƯỚC 2: LOAD VÀ TÍNH TOÁN ĐẶC TRƯNG VẬT LÝ
# ==========================================
# Giả lập dữ liệu (14 cột)
columns_list = [
    'time', 'roll', 'pitch', 'yaw', 
    'p', 'q', 'r', 
    'V_body_x', 'V_body_y', 'V_body_z', 
    'm1', 'm2', 'm3', 'm4'
]

# Tạo dữ liệu ảo để test code (Thay đoạn này bằng pd.read_csv('AI_Training_Data.csv') khi chạy thật)
np.random.seed(42)
raw_data = np.random.rand(2000, 14) 
df = pd.DataFrame(raw_data, columns=columns_list)

# --- TÍNH TOÁN MOMEN (TAU) THEO CÔNG THỨC CHỮ X ---
# Lực đẩy tỷ lệ thuận với bình phương tốc độ quay
m1_sq = df['m1']**2
m2_sq = df['m2']**2
m3_sq = df['m3']**2
m4_sq = df['m4']**2

# Tính Momen giả định (Pseudo-Torques) cho Roll và Pitch
df['tau_roll']  = m2_sq + m3_sq - m1_sq - m4_sq
df['tau_pitch'] = m2_sq + m4_sq - m1_sq - m3_sq

# ==========================================
# BƯỚC 3: LỌC NHIỄU VÀ CHUẨN HÓA (Z-SCORE)
# ==========================================
# Kiến trúc mới: Input X chỉ có 2 Momen, Output Y chỉ có 2 Góc
cols_X = ['tau_roll', 'tau_pitch']
cols_Y = ['roll', 'pitch']

# Lọc nhiễu Butterworth cho tín hiệu Momen (Vì lệnh động cơ m1..m4 đôi khi bị giật cục)
FS = 100.0 
CUTOFF = 15.0 
df[cols_X] = butter_lowpass_filter(df[cols_X].values, CUTOFF, FS)

# Chuẩn hóa để AI hội tụ nhanh
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

normalized_X = scaler_X.fit_transform(df[cols_X].values)
normalized_Y = scaler_Y.fit_transform(df[cols_Y].values)

# ==========================================
# BƯỚC 4: CẮT CỬA SỔ TRƯỢT (SLIDING WINDOW)
# ==========================================
def create_sliding_windows(X_data, Y_data, window_size):
    """
    Dùng chuỗi lực tác động từ t-n đến t (độ dài window_size) để dự đoán Góc tại t+1.
    """
    X, Y = [], []
    for i in range(len(X_data) - window_size):
        window_x = X_data[i : i + window_size, :]  # Quá khứ: Lấy 2 cột Tau
        X.append(window_x) 

        target_y = Y_data[i + window_size, :]      # Tương lai: Lấy 2 cột Góc roll , pitch
        Y.append(target_y)

    return np.array(X), np.array(Y)

WINDOW_SIZE = 50 # 0.5 giây ở tần số 100Hz
X_full, Y_full = create_sliding_windows(normalized_X, normalized_Y, WINDOW_SIZE)

# ==========================================
# BƯỚC 5: CHIA TẬP TRAIN/VAL/TEST (CHRONOLOGICAL)
# ==========================================
# Tuyệt đối không xáo trộn (shuffle) dữ liệu chuỗi thời gian
total_samples = len(X_full)
train_end = int(total_samples * 0.70) # 70% Train
val_end   = int(total_samples * 0.85) # 15% Val, 15% Test

X_train, Y_train = X_full[:train_end], Y_full[:train_end]
X_val, Y_val     = X_full[train_end:val_end], Y_full[train_end:val_end]
X_test, Y_test   = X_full[val_end:], Y_full[val_end:]

# ==========================================
# IN KIỂM TRA KẾT QUẢ
# ==========================================
print("--- CẤU TRÚC DỮ LIỆU ĐÃ SẴN SÀNG CHO PYTORCH ---")
print(f"Kích thước X_train: {X_train.shape} -> (Batch, {WINDOW_SIZE} timesteps, 2 features: tau_roll, tau_pitch)")
print(f"Kích thước Y_train: {Y_train.shape} -> (Batch, 2 features: roll, pitch)")
print(f"Phân bổ dữ liệu: Train = {len(X_train)} mẫu | Val = {len(X_val)} mẫu | Test = {len(X_test)} mẫu")