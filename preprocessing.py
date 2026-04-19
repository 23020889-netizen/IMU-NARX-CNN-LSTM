import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# Step 1: low pass filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    # fs: sampling frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

# Step 2: load and preprocess data
# suppose: data in CSV format, with columns: ... 
# make simulation data to test the code
columns_list = [
    'roll', 'pitch', 'yaw', 
    'p', 'q', 'r', 
    'Vbx', 'Vby', 'Vbz', 
    'm1', 'm2', 'm3', 'm4'
]

np.random.seed(42)
raw_data = np.random.rand(2000,13) # simulate raw data
df = pd.DataFrame(raw_data, columns=columns_list)

FS = 100.0 # sampling frequency
CUTOFF = 15.0 # filter the oscillations above 15 Hz 

# Chỉ lọc nhiễu các tín hiệu vận tốc góc và vận tốc thân, 
# các góc yaw, pitch, roll và tín hiệu động cơ thường đã mượt
noisy_cols = ['p', 'q', 'r', 'Vbx', 'Vby', 'Vbz']
df[noisy_cols] = butter_lowpass_filter(df[noisy_cols].values, CUTOFF, FS)

# Step 3: Z score normalization
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Deep learning models require normalized data with average of 0 and standard deviation of 1
# Input X bao gồm cả 13 cột)
cols_X = columns_list
# Output Y chỉ có 3 cột: yaw, pitch, roll
cols_Y = ['roll', 'pitch', 'yaw']
normalized_X = scaler_X.fit_transform(df[cols_X].values)
normalized_Y = scaler_Y.fit_transform(df[cols_Y].values)

# Step 4: Sliding window segmentation

def create_sliding_windows(X_data, Y_data, window_size):
    """
    Cắt chuỗi thời gian bằng cửa sổ trượt.
    Dùng dữ liệu từ t-n đến t (độ dài window_size) để dự đoán Y tại t+1.
    """
    X = []
    Y = []

    for i in range(len(X_data) - window_size):
        # Đặc trưng X (Quá khứ): Từ thời điểm i đến i + window_size (Gồm cả 13 cột)
        window_x = X_data[i: i + window_size, :]  # collect all columns for features
        X.append(window_x)  # collect all columns for features

        # Label Y (Target): time: t + 1 (only 3 colums: yaw, pitch, roll
        target_y = Y_data[i + window_size, :] 
        Y.append(target_y)

    return np.array(X), np.array(Y)

# Configure window
WINDOW_SIZE = 50 # 0.5 seconds at 100 Hz
X_train, Y_train = create_sliding_windows(normalized_X, normalized_Y, WINDOW_SIZE)

# Test output shapes
print(f"Kích thước X_train (Inputs): {X_train.shape} -> (Số mẫu, {WINDOW_SIZE} time-steps, 13 features)")
print(f"Kích thước Y_train (Labels): {Y_train.shape} -> (Số mẫu, 3 features (yaw, pitch, roll))")

