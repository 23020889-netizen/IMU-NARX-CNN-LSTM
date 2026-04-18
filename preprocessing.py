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
    y = filtfilt(b, a, data)
    return y

# Step 2: load and preprocess data
# suppose: data in CSV format, with columns: ... 
# make simulation data to test the code
np.random.seed(42)
raw_data = np.random.rand(1000,10) # simulate raw data

FS = 100.0 # sampling frequency
CUTOFF = 10.0 # filter the oscillations above 10 Hz 

# Apply low pass filter to each column of the data (13 columns)
# 4 monitoring columns no filter needed
filtered_imu = butter_lowpass_filter(raw_data[:, :9], CUTOFF, FS)
processed_data = np.hstack((filtered_imu, raw_data[:, 9:])) # concat with motor data

# Step 3: Z score normalization
scaler = StandardScaler()
# Deep learning models require normalized data with average of 0 and standard deviation of 1
normalized_data = scaler.fit_transform(processed_data)

# Step 4: Sliding window segmentation

def create_sliding_windows(data, window_size, target_indices):
    # NARX 
    X = []
    Y = []

    for i in range(len(data) - window_size):
        # Đặc trưng X (Quá khứ): Từ thời điểm i đến i + window_size (Gồm cả 13 cột)
        window_x = data[i:i + window_size, :]  # collect all columns for features
        X.append(window_x)  # collect all columns for features

        # Label Y (Target): time: t + 1 (only 3 colums: yaw, pitch, roll
        # target_y = data[i + window_size, target_indices]  # collect only target columns for labels
        target_y = data[i + window_size, target_indices] 
        Y.append(target_y)

    return np.array(X), np.array(Y)

# Configure window
WINDOW_SIZE = 50 # 0.5 seconds at 100 Hz
TARGET_INDICES = [9, 10, 11] # yaw, pitch, roll
X_train, Y_train = create_sliding_windows(normalized_data, WINDOW_SIZE, TARGET_INDICES)

print(f"Kích thước X ban đầu (Raw data): {raw_data.shape}")
print(f"Kích thước tensor X đầu vào AI: {X_train.shape} --> (Số mẫu, Cửa sổ thời gian, Số tính năng)")
print(f"Kích thước tensor Y nhãn AI: {Y_train.shape} --> (Số mẫu, Giá trị IMU dự đoán)")

