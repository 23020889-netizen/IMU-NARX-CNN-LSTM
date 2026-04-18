# IMU-NARX-CNN-LSTM
Bước 1: Chuẩn bị và phân mảnh dữ liệu
    Sử dụng các công cụ:
    - Filtering: Moving Average Filter, Band-pass/Low-pass Filter: loại bỏ nhiễu tần số cao, làm nổi bật xu hướng chuyển động
    - Normalization: Tốc độ động cơ rất lớn, trong khi vận tốc góc nhỏ, nếu để nguyên mạng AI sẽ bị chệch hướng, khong hội tụ được. Phương pháp sử dụng: Z-score normalization: chuyển all kênh dữ liệu về dạng phân bố chuẩn: giá trị trung bình = 0 và độ lệch chuẩn = 1.
    - Thư viện:
        + numpy và pandas: Để tải, thao tác và tính toán ma trận dữ liệu
        + scipy: Thiết kế các bộ lọc nhiễu
        + scikit-learn: Cung cấp sẵn hàm StandardScaler để tự động chuẩn hóa Z-score.
