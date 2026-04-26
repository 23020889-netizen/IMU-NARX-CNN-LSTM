import torch
import torch.nn as nn

class UAV_Torque_Net(nn.Module):
    def __init__(self, seq_length=20):
        super(UAV_Torque_Net, self).__init__()
        
        # ----------------------------------------------------
        # 1. KHỐI CNN 1D: "ÉP CÂN" THAM SỐ (Channels 16 -> 32)
        # ----------------------------------------------------
        self.cnn_block = nn.Sequential(
            # Layer 1: Giảm từ 32 xuống 16 channels
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            
            # Layer 2: Giảm từ 64 xuống 32 channels
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) 
        )
        
        self.cnn_out_length = seq_length // 4  
        
        # ----------------------------------------------------
        # 2. KHỐI LSTM: GIẢM HIDDEN SIZE & TĂNG DROPOUT
        # ----------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=32,        # Phải khớp với out_channels của CNN mới (32)
            hidden_size=64,       # Giảm từ 128 xuống 64 để giảm sức chứa "bộ nhớ vẹt"
            num_layers=2,         
            batch_first=True,     
            dropout=0.4           # Tăng từ 0.2 lên 0.4 để ép mạng "phải quên" nhiễu
        )
        
        # ----------------------------------------------------
        # 3. KHỐI DENSE (LINEAR): THU GỌN LỚP ẨN
        # ----------------------------------------------------
        self.fc_block = nn.Sequential(
            nn.Linear(64, 32),    # Đầu vào 64 khớp với hidden_size của LSTM
            nn.ReLU(),
            nn.Dropout(0.4),      # Tăng Dropout ở lớp cuối để tăng tính ổn định
            nn.Linear(32, 2)      # Đầu ra: (Roll, Pitch)
        )

    def forward(self, x):
        # Chuyển từ (Batch, Seq, Feat) -> (Batch, Feat, Seq) cho CNN
        x = x.permute(0, 2, 1)
        x = self.cnn_block(x) 
        # Chuyển lại cho LSTM: (Batch, Seq_new, Feat_new)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Lấy trạng thái ở timestep cuối cùng
        last_time_step_out = lstm_out[:, -1, :] 
        predictions = self.fc_block(last_time_step_out)
        return predictions