import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu đầu vào
models = ["ResNet1D", "CNN_GRU", "BiMamba"]
datasets = ["XRF55", "ESP", "ASUS"]

result = {
    "case0": {
        "XRF55": [21.77, 29.38, 26.88],
        "ESP":   [70.78, 31.41, 37.58],
        "ASUS":  [48.83, 37.97, 60.86],
    },
    "case1": {
        "XRF55": [40.52, 43.23, 42.92],
        "ESP":   [81.02, 71.41, 73.67],
        "ASUS":  [73.83, 60.86, 68.91],
    },
    "case2": {
        "XRF55": [44.17, 46.15, 46.46],
        "ESP":   [67.27, 52.89, 55.16],
        "ASUS":  [60.94, 55.47, 64.92],
    },
    "case3": {
        "XRF55": [44.8, 42.40, 49.48],
        "ESP":   [87.03, 75.08, 68.67],
        "ASUS":  [82.97, 77.81, 80.55],
    },
    "case4": {
        "XRF55": [39.27, 40.73, 41.25],
        "ESP":   [74.14, 61.48, 69.69],
        "ASUS":  [76.56, 63.05, 75.78],
    }
}

# Khởi tạo dictionary để lưu kết quả trung bình và độ lệch chuẩn
means = {ds: [] for ds in datasets}
stds = {ds: [] for ds in datasets}

# Tính toán Mean và Standard Deviation cho 4 cases
for ds in datasets:
    for i in range(len(models)):
        acc_values = [result[case][ds][i] for case in ["case0", "case1", "case2", "case3", "case4"]]
        means[ds].append(np.mean(acc_values))
        stds[ds].append(np.std(acc_values, ddof=1))

# Cấu hình vẽ biểu đồ
plt.figure(figsize=(12, 7))  # Mở rộng nhẹ chiều ngang đồ thị để lấy chỗ trống hiển thị Text

# Chuyển đổi nhãn trục X thành các vị trí số nguyên để dễ dàng dịch chuyển
x = np.arange(len(models))

# Đặt độ lệch (offset) cho 3 dataset: Trái, Giữa, Phải
offsets = [-0.1, 0, 0.1] 

markers = ['o', 's', '^']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Vẽ 3 đường đồ thị với tọa độ X đã được dịch chuyển nhẹ
for idx, ds in enumerate(datasets):
    x_pos = x + offsets[idx]
    
    # Vẽ đường và thanh râu sai số
    plt.errorbar(
        x_pos, 
        means[ds], 
        yerr=stds[ds], 
        label=ds, 
        marker=markers[idx], 
        color=colors[idx], 
        capsize=5, 
        capthick=2, 
        linewidth=2, 
        markersize=8
    )
    
    # Thêm text hiển thị giá trị trung bình tại từng điểm dữ liệu
    for i in range(len(models)):
        plt.text(
            x_pos[i] + 0.02,          # Đẩy text sang phải một chút so với điểm marker
            means[ds][i] + 1.0,       # Đẩy text lên trên một chút cho dễ nhìn
            f'{means[ds][i]:.2f}',    # Format số hiển thị 2 chữ số thập phân
            color=colors[idx],
            fontsize=9,
            ha='left',
            va='bottom'
        )

# Đặt lại nhãn cho trục X bằng tên các mô hình
plt.xticks(x, models)

# Căn chỉnh hiển thị
plt.title('Độ chính xác trung bình của các mô hình trên 3 Datasets (4 Cases)', fontsize=14, pad=15)
plt.xlabel('Tên mô hình', fontsize=12)
plt.ylabel('Độ chính xác trung bình (%)', fontsize=12)
plt.legend(title='Dataset', fontsize=10, title_fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Lưu và hiển thị biểu đồ
plt.savefig('accuracy_models_comparison_offset_with_text.png', dpi=300)
plt.show()