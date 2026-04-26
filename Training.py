import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random

# Tích hợp biến từ 2 file trên
from CNN_LSTM import UAV_Torque_Net
from preprocessing import X_train, Y_train, X_val, Y_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

BATCH_SIZE = 64
# BẬT SHUFFLE = TRUE ĐỂ CHỐNG KẸT NGHIỆM
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, Y_val_tensor), batch_size=BATCH_SIZE, shuffle=False)

# Đổi từ 50 xuống 20
model = UAV_Torque_Net(seq_length=20).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 100
best_val_loss = float('inf')
patience_early_stop = 15
trigger_times = 0

# Biến Scheduled Sampling (Tỷ lệ mớm bài giải)
teacher_forcing_ratio = 1.0 

for epoch in range(epochs):
    model.train() 
    train_loss = 0.0
    
    # ÉP MÔ HÌNH KHÔNG Ỷ LẠI: Cứ 10 epoch giảm tỷ lệ mớm đi 10%
    if epoch > 0 and epoch % 10 == 0:
        teacher_forcing_ratio = max(0.5, teacher_forcing_ratio - 0.1)
        print(f"📉 Teacher Forcing Ratio giảm còn: {teacher_forcing_ratio:.1f}")

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # --- LÕI CỦA SCHEDULED SAMPLING ---
        # Chỉ mục cột: 0=tau_roll, 1=tau_pitch, 2=roll, 3=pitch
        # Nếu rơi vào xác suất không được "mớm" bài chuẩn, ta bơm nhiễu vào cột 2 và 3
        if random.random() > teacher_forcing_ratio:
            noise = torch.randn_like(inputs[:, :, 2:4]) * 0.05 # Lệch khoảng 0.05 radian (~3 độ)
            inputs[:, :, 2:4] += noise
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()
            
    avg_val_loss = val_loss / len(val_loader)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{epochs}] | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}')
    if optimizer.param_groups[0]['lr'] < current_lr:
        print(f"   ⚠️ Giảm Learning Rate xuống {optimizer.param_groups[0]['lr']}")
    
    # EARLY STOPPING
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), '/kaggle/working/uav_torque_best.pth') 
    else:
        trigger_times += 1
        if trigger_times >= patience_early_stop:
            print(f"🛑 Dừng sớm tại Epoch {epoch+1}. Best Val MSE: {best_val_loss:.6f}")
            break