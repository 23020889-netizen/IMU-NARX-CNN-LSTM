import torch.optim as optim

# Khởi tạo mô hình
model = UAV_Torque_Net(seq_length=50)

# Khai báo thiết bị (Sử dụng GPU nếu có để train nhanh hơn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 1. Hàm Loss: MSE để hội tụ chính xác quỹ đạo
criterion = nn.MSELoss()

# 2. Optimizer: Adam tự điều chỉnh learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # weight_decay giúp Regularization