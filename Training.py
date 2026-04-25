epochs = 100

for epoch in range(epochs):
    model.train() # Chuyển sang chế độ train (Kích hoạt Dropout & BatchNorm)
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        # inputs shape: (Batch, 50, 2) - Gồm tau_roll, tau_pitch
        # targets shape: (Batch, 2) - Gồm roll_true, pitch_true
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Bước 1: Xóa gradient cũ
        optimizer.zero_grad()
        
        # Bước 2: Truyền tiến (Forward pass)
        outputs = model(inputs)
        
        # Bước 3: Tính sai số (MSE Loss)
        loss = criterion(outputs, targets)
        
        # Bước 4: Lan truyền ngược (Backpropagation)
        loss.backward()
        
        # Kỹ thuật Gradient Clipping (Cực kỳ quan trọng cho LSTM để chống nổ Gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Bước 5: Cập nhật trọng số
        optimizer.step()
        
        train_loss += loss.item()
        
    # Tính Loss trung bình của cả epoch
    avg_train_loss = train_loss / len(train_loader)
    
    # ----------------------------------------------------
    # QUÁ TRÌNH VALIDATION (Kiểm tra tránh học vẹt)
    # ----------------------------------------------------
    model.eval() # Tắt Dropout để test
    val_loss = 0.0
    with torch.no_grad(): # Không lưu đồ thị tính toán để tiết kiệm RAM
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_targets)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch [{epoch+1}/{epochs}] | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}')