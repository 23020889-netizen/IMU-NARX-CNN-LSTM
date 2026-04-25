import torch
import torch.nn as nn

class UAV_Torque_Net(nn.Module):
    def __init__(self, seq_length=50):
        super(UAV_Torque_Net, self).__init__()
        
        # ----------------------------------------------------
        # 1. KHỐI CNN 1D: Trích xuất đặc trưng Momen cục bộ
        # Đầu vào: (Batch, 2 kênh Momen, Chiều dài chuỗi L)
        # ----------------------------------------------------
        self.cnn_block = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Nén chiều dài chuỗi đi một nửa
            
            # Layer 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Tiếp tục nén
        )
        
        # Tính toán chiều dài chuỗi còn lại sau 2 lần MaxPool
        # Ví dụ: Đầu vào L=50 -> L=25 -> L=12
        self.cnn_out_length = seq_length // 4  
        
        # ----------------------------------------------------
        # 2. KHỐI LSTM: Học quán tính và sự phụ thuộc thời gian
        # ----------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=64,        # Bằng với out_channels của khối CNN cuối cùng
            hidden_size=128,      # Sức chứa bộ nhớ quán tính
            num_layers=2,         # 2 lớp LSTM chồng lên nhau để học sâu hơn
            batch_first=True,     # Giữ Batch_size ở chiều đầu tiên
            dropout=0.2           # Chống học vẹt (Overfitting)
        )
        
        # ----------------------------------------------------
        # 3. KHỐI DENSE (LINEAR): Ánh xạ ra Góc dự đoán
        # ----------------------------------------------------
        self.fc_block = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)      # Đầu ra: 2 giá trị dự đoán (Roll, Pitch)
        )

    def forward(self, x):
        # x ban đầu thường có dạng: (Batch, Seq_Length, Features=2)
        # Bắt buộc phải đảo trục để đưa vào Conv1d: (Batch, Channels, Seq_Length)
        x = x.permute(0, 2, 1)
        
        # Đi qua khối CNN
        x = self.cnn_block(x) 
        # Output x lúc này: (Batch, 64 channels, cnn_out_length)
        
        # Đảo trục lại để đưa vào LSTM: (Batch, Seq_Length_New, Features=64)
        x = x.permute(0, 2, 1)
        
        # Đi qua khối LSTM
        # out chứa toàn bộ trạng thái; (h_n, c_n) chứa trạng thái cuối cùng
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Trích xuất đầu ra ở bước thời gian CUỐI CÙNG của chuỗi
        # Kích thước: (Batch, 128)
        last_time_step_out = lstm_out[:, -1, :] 
        
        # Đi qua lớp Fully Connected để ra góc dự đoán
        # Kích thước: (Batch, 2)
        predictions = self.fc_block(last_time_step_out)
        
        return predictions