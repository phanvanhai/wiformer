# Tổng hợp: Project Benchmark Mô hình WiFi Sensing – HAR

Tài liệu này tổng hợp lại toàn bộ yêu cầu, quyết định thiết kế, cấu trúc project và
các lưu ý kỹ thuật đã thống nhất qua cuộc trao đổi.

---

## 1. Bài toán & Yêu cầu ban đầu

- **Mục tiêu:** Viết code benchmark nhiều mô hình cho bài toán **WiFi Sensing – Human Activity Recognition (HAR)**.
- **7 mô hình cần benchmark:**
  1. MLP
  2. CNN-5
  3. BiLSTM
  4. ViT
  5. ResNet18
  6. CNN + GRU
  7. BiMamba
- **3 bộ dữ liệu:**
  1. `UT_HAR`
  2. `SSHAR_Nexmon`
  3. `SSHAR_ESP`
  - Tỉ lệ chia **train/test = 80/20**.
- **Đầu ra cần lưu cho mỗi tổ hợp (dataset × model):**
  - Đường cong train (learning curve)
  - Đường cong test
  - Ma trận nhầm lẫn (confusion matrix)
  - Giá trị Accuracy, F1

---

## 2. Quá trình thiết kế cấu trúc project

### 
**Lưu ý kỹ thuật theo từng model (ở bản đầu):**
- **MLP / CNN-5:** input cần flatten hoặc reshape về dạng ảnh 2D (time × subcarrier).
- **ResNet18:** thường cần input 3 kênh — có thể duplicate/stack CSI amplitude thành pseudo-RGB, hoặc sửa `conv1` để nhận 1 kênh.
- **ViT:** cần patch embedding phù hợp kích thước CSI (không phải ảnh tự nhiên); nên dùng ViT nhỏ (patch size nhỏ, ít layer) để tránh overfit do dữ liệu CSI ít hơn ImageNet.
- **BiLSTM / CNN+GRU:** input dạng sequence (time, feature).
- **BiMamba:** cần cài `mamba-ssm` (yêu cầu CUDA), hoặc dùng implementation thuần PyTorch nếu không có GPU phù hợp — nên tách riêng file `bimamba.py` với `try/except` import để không phá vỡ toàn bộ pipeline khi thiếu dependency.

### 2.2 Phiên bản đơn giản hoá (theo yêu cầu người dùng: "cấu trúc hơi phức tạp, đơn giản hóa lại")

```
wifi-har-benchmark/
├── data/
│   ├── UT_HAR/
│   ├── SSHAR_Nexmon/
│   └── SSHAR_ESP/
│
├── datasets.py     # load + tiền xử lý + split 80/20 cho cả 3 dataset
├── models.py       # định nghĩa cả 7 model (MLP, CNN5, BiLSTM, ViT, ResNet18, CNN_GRU, BiMamba)
├── train.py        # vòng lặp train/test chung, dùng cho mọi model/dataset
├── metrics.py       # tính Acc, F1, confusion matrix + vẽ biểu đồ
├── main.py          # chạy benchmark: loop qua 3 dataset x 7 model
│
├── results/
│   └── {dataset}_{model}/
│       ├── history.csv          # loss/acc train & test theo epoch
│       ├── metrics.json          # Acc, F1 cuối cùng
│       ├── confusion_matrix.png
│       └── learning_curve.png
│
├── requirements.txt
└── README.md
```

**Giải thích từng file (5 file code chính):**

- **`datasets.py`**
  ```python
  def load_dataset(name):   # name: "ut_har" | "sshar_nexmon" | "sshar_esp"
      # đọc raw data, tiền xử lý, trả về X, y

  def split_data(X, y, ratio=0.8, seed=42):
      # train_test_split, trả về train_loader, test_loader
  ```

- **`models.py`**
  ```python
  def get_model(name, input_shape, num_classes):
      # name: "mlp" | "cnn5" | "bilstm" | "vit" | "resnet18" | "cnn_gru" | "bimamba"
      # trả về model tương ứng (mỗi model là 1 class nhỏ trong cùng file)
  ```

- **`train.py`**
  ```python
  def train_and_evaluate(model, train_loader, test_loader, epochs):
      # train, ghi lại loss/acc mỗi epoch (train + test)
      # sau khi train xong -> chạy evaluate trên test set
      # trả về history (dict) + kết quả cuối (y_true, y_pred)
  ```

- **`metrics.py`**
  ```python
  def compute_metrics(y_true, y_pred):
      # trả về acc, f1, confusion_matrix

  def plot_learning_curve(history, save_path): ...
  def plot_confusion_matrix(cm, save_path): ...
  ```

- **`main.py`**
  ```python
  datasets = ["ut_har", "sshar_nexmon", "sshar_esp"]
  models = ["mlp", "cnn5", "bilstm", "vit", "resnet18", "cnn_gru", "bimamba"]

  for ds_name in datasets:
      X, y = load_dataset(ds_name)
      train_loader, test_loader = split_data(X, y)
      for model_name in models:
          model = get_model(model_name, input_shape, num_classes)
          history, y_true, y_pred = train_and_evaluate(model, train_loader, test_loader, epochs)
          save_dir = f"results/{ds_name}_{model_name}"
          os.makedirs(save_dir, exist_ok=True)
          acc, f1, cm = compute_metrics(y_true, y_pred)
          save_json({"acc": acc, "f1": f1}, f"{save_dir}/metrics.json")
          save_csv(history, f"{save_dir}/history.csv")
          plot_learning_curve(history, f"{save_dir}/learning_curve.png")
          plot_confusion_matrix(cm, f"{save_dir}/confusion_matrix.png")
  ```

**Ưu điểm của bản đơn giản hoá:**
- Chỉ 5 file code chính thay vì hàng chục file rải rác → dễ đọc, dễ debug, dễ sửa nhanh khi thử nghiệm.
- Không dùng YAML config phức tạp — tham số (epochs, lr, batch_size...) để thẳng trong `main.py` hoặc argparse đơn giản.
- Vẫn giữ nguyên tắc: tách biệt data / model / train / metrics, kết quả từng tổ hợp lưu riêng thư mục để dễ so sánh.
- Mở rộng (thêm model, thêm dataset) chỉ cần thêm 1 hàm trong `models.py` / `datasets.py`.

---

## 3. Chuẩn hoá lưu trữ History để vẽ nhiều model trên 1 đồ thị

**Yêu cầu người dùng:** đường cong lưu ra CSV để sau này vẽ nhiều mô hình trên cùng 1 đồ thị.

### 3.1 Format `history.csv` (long format, mỗi dòng = 1 epoch)

Có thêm cột `dataset` và `model` để sau này gộp nhiều file mà không lẫn lộn:

| epoch | train_loss | train_acc | test_loss | test_acc | dataset | model |
|-------|-----------|-----------|-----------|----------|---------|-------|
| 1     | 1.82      | 0.31      | 1.65      | 0.35     | ut_har  | mlp   |
| 2     | 1.40      | 0.48      | 1.30      | 0.50     | ut_har  | mlp   |
| ...   | ...       | ...       | ...       | ...      | ...     | ...   |

Ghi ra `results/{dataset}_{model}/history.csv` sau mỗi lần train.

### 3.2 File tổng hợp `results/all_history.csv`

Ngoài file riêng từng tổ hợp, **append** thêm vào 1 file chung để không phải đọc từng file lẻ khi vẽ so sánh:

```python
all_history_path = "results/all_history.csv"
if os.path.exists(all_history_path):
    df_history.to_csv(all_history_path, mode="a", header=False, index=False)
else:
    df_history.to_csv(all_history_path, mode="w", header=True, index=False)
```

### 3.3 Script vẽ lại độc lập — `plot_compare.py`

```python
def plot_compare(dataset_name, models, metric="test_acc", save_path=None):
    df = pd.read_csv("results/all_history.csv")
    df = df[df["dataset"] == dataset_name]
    plt.figure(figsize=(8, 5))
    for model_name in models:
        sub = df[df["model"] == model_name]
        plt.plot(sub["epoch"], sub[metric], label=model_name)
    plt.xlabel("Epoch"); plt.ylabel(metric)
    plt.title(f"{metric} on {dataset_name}")
    plt.legend(); plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
```

→ Có thể so sánh acc/loss, train/test, hoặc nhiều dataset chỉ bằng cách đổi tham số `metric` / `dataset_name`, **không cần train lại**.

### 3.4 Cấu trúc `results/` sau cập nhật

```
results/
├── all_history.csv        # toàn bộ 21 tổ hợp, dùng để vẽ so sánh
├── ut_har_mlp/
│   ├── history.csv
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── learning_curve.png
├── ut_har_cnn5/
│   └── ...
...
plot_compare.py             # vẽ lại bất cứ lúc nào, độc lập với lúc train
```

---

## 4. Làm rõ yêu cầu (Requirements Clarification)

Người dùng xác nhận 4 yêu cầu bổ sung, chưa vội code:

1. **Training:** tối đa **100 epoch/model**, có thể có early-stopping hoặc không, nhưng **quan trọng nhất là theo dõi metric tốt nhất** (thường là `test_acc` cao nhất hoặc `val_loss` thấp nhất) qua các epoch → lưu checkpoint tại epoch đó (`best_model.pth`), **không phải checkpoint epoch cuối cùng**.
2. **Best epoch** cần được đánh dấu trong `metrics.json` (ví dụ: `best_epoch: 42`) và dùng chính model tại epoch đó để tính Acc/F1/confusion matrix cuối cùng (không phải model ở epoch 100).
3. **Path compatibility Windows ↔ Kaggle:** code viết trên Windows (đường dẫn `\`) nhưng chạy trên Kaggle (Linux, đường dẫn `/`, thư mục input thường là `/kaggle/input/...`) → cần dùng `pathlib.Path` hoặc `os.path.join` xuyên suốt, **tuyệt đối không hardcode `\` hoặc nối chuỗi path bằng `+`**.
4. **Đếm tổng tham số model** → cần thư viện in ra param count (và tốt hơn là FLOPs) để so sánh độ phức tạp giữa 7 model.

### 4.1 Đếm tổng tham số model — **Phương án đã chọn: `torchinfo` (giống Keras `model.summary()`)**

Dùng `torchinfo` để in bảng tổng quan theo từng layer (tên layer, output shape, param count), tương tự `model.summary()` của Keras, đồng thời lấy được tổng số tham số để lưu vào `metrics.json`:

```python
from torchinfo import summary

model_stats = summary(
    model,
    input_size=(batch_size, *input_shape),
    verbose=0,   # verbose=0 để không in ra console, chỉ lấy object kết quả
)

total_params = model_stats.total_params
trainable_params = model_stats.trainable_params
```

- In bảng đẹp ra màn hình/log khi cần debug: `print(model_stats)`.
- Vì mỗi dataset có `input_shape` khác nhau (UT_HAR / SSHAR-ESP / SSHAR-Nexmon), **phải gọi `summary()` riêng cho từng cặp (model, dataset)**, không dùng chung 1 `input_shape` cố định.
- Lưu `total_params` và `trainable_params` vào `metrics.json` của từng tổ hợp (dataset, model) để khi tổng hợp báo cáo có sẵn cột "Params" so sánh cùng Acc/F1.
- **Lưu ý rủi ro:** với model dạng sequence/mamba (BiMamba, có thể cả BiLSTM/CNN+GRU) `torchinfo` đôi khi gặp lỗi do input shape đặc biệt hoặc custom CUDA kernel. Nên bọc lệnh gọi trong `try/except`, và nếu lỗi thì fallback về cách đếm thủ công không phụ thuộc thư viện ngoài:

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

→ Áp dụng: `torchinfo` là cách chính thức để lấy `total_params`/`trainable_params` và bảng summary layer-by-layer; `count_parameters()` (thủ công) dùng làm phương án dự phòng khi `torchinfo` lỗi với kiến trúc đặc biệt (đặc biệt BiMamba).

*(FLOPs qua `thop`/`fvcore` không được chọn dùng do rủi ro không tương thích với layer custom của BiMamba.)*

### 4.2 Nguyên tắc path compatibility (áp dụng xuyên suốt code)

```python
from pathlib import Path

# Không viết: "data/UT_HAR/train.csv" nối bằng + hoặc f-string thô với "\"
# Luôn viết:
DATA_ROOT = Path("data")   # hoặc Path("/kaggle/input/...") khi chạy Kaggle
data_path = DATA_ROOT / "UT_HAR" / "train.csv"

RESULTS_ROOT = Path("results")
save_dir = RESULTS_ROOT / f"{dataset_name}_{model_name}"
save_dir.mkdir(parents=True, exist_ok=True)
```

→ Nên có 1 biến `DATA_ROOT` / `RESULTS_ROOT` khai báo đầu file (hoặc check `os.path.exists("/kaggle")` để tự động switch), để không phải sửa tay mỗi lần chuyển máy.

---

## 5. Thiết kế Dataset Loader thống nhất (dựa trên code người dùng cung cấp)

### 5.1 Hiểu về cách đọc dữ liệu gốc

- **UT_HAR:** dữ liệu đã được load sẵn toàn bộ vào RAM dưới dạng file `.csv` (thực chất là `.npy` lưu đuôi `.csv`) cho train/val/test + label tương ứng, reshape về `(N, 90, 250)`, **normalize min-max theo từng file** (không theo global train).
- **SSHAR (ESP/Nexmon):** dữ liệu là hàng ngàn file `.npy` rời rạc theo cấu trúc thư mục `room/device/rx/subject/act_pos_dir_rep.npy`. Dataset build 1 index (danh sách sample) dựa trên tên file, chia train/test theo `rep` (rep ≤ 8 với `dir=0`, rep ≤ 4 với `dir≠0` → xấp xỉ train/test), chuẩn hoá **z-score** bằng mean/std tính riêng trên tập train, sau đó pool thời gian 1000→250 để khớp chiều với UT_HAR (90 hoặc 168/672 kênh × 250 timestep).

### 5.2 Các điểm cần thống nhất (đã được confirm/điều chỉnh theo phản hồi người dùng)

| # | Vấn đề | Đề xuất ban đầu | **Quyết định cuối (theo yêu cầu người dùng)** |
|---|--------|------------------|-----------------------------------------------|
| 1 | UT_HAR vốn đã có sẵn train/val/test theo folder | Gộp hết train+val+test rồi tự chia lại 80/20 giống SSHAR | **Giữ nguyên split gốc theo folder** (train/val/test đã có sẵn), **không** gộp rồi chia lại 80/20 |
| 2 | SSHAR ESP và Nexmon | Dùng chung 1 hàm/class với tham số `device` | **Viết 2 hàm/class riêng biệt** cho ESP và Nexmon (API bên ngoài là 2 tên riêng), nhưng bên trong vẫn gọi 1 hàm dùng chung (`_SSHARDatasetBase`) để đỡ lặp code |
| 3 | Chuẩn hoá khác kiểu nhau (UT_HAR: min-max theo mảng; SSHAR: z-score theo train) | Thống nhất z-score cho cả 3 dataset | **Đưa thành tham số tùy chọn**: `normalize` (bool) + `norm_type` (`'minmax'` hoặc `'zscore'`) — để người dùng bật/tắt hoặc đổi kiểu lúc gọi, không hardcode |
| 4 | Input shape khác nhau (UT_HAR: 90×250, SSHAR-ESP: 168×250, SSHAR-Nexmon: 672×250) | Mỗi model cần biết `input_shape` để khởi tạo đúng | `main.py` tự lấy `input_shape` từ 1 sample thay vì hardcode |

### 5.3 Cấu trúc code cuối cùng (bản chỉnh theo yêu cầu)

**`UTHARDataset`** — dùng split có sẵn theo folder:

```python
class UTHARDataset(Dataset):
    """
    root_dir/UT_HAR/data/{train,val,test}_data.csv
    root_dir/UT_HAR/label/{train,val,test}_label.csv
    """
    def __init__(self, root_dir, split='train', normalize=True, norm_type='minmax',
                 mean=None, std=None):
        ...
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.normalize:
            if self.norm_type == 'minmax':
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            elif self.norm_type == 'zscore':
                assert self.mean is not None and self.std is not None
                x = (x - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(x).float(), int(self.y[idx])

def compute_ut_har_mean_std(root_dir, split='train'):
    """Chỉ cần dùng khi norm_type='zscore'."""
    ...
```

**SSHAR — core dùng chung (private), rồi expose 2 class riêng:**

```python
class _SSHARDatasetBase(Dataset):
    def __init__(self, root_dir, device, signal='amp', split='train',
                 normalize=True, norm_type='zscore', mean=None, std=None):
        ...
        if self.normalize and self.norm_type == 'zscore' and (mean is None or std is None):
            raise ValueError("norm_type='zscore' cần truyền mean/std (tính từ tập train)")
        ...

# ---------- ESP ----------
class SSHAR_ESP_Dataset(_SSHARDatasetBase):
    def __init__(self, root_dir, signal='amp', split='train',
                 normalize=True, norm_type='zscore', mean=None, std=None):
        super().__init__(root_dir, device='esp', signal=signal, split=split,
                          normalize=normalize, norm_type=norm_type, mean=mean, std=std)

def compute_sshar_esp_mean_std(root_dir, signal='amp'):
    return _compute_sshar_mean_std(root_dir, device='esp', signal=signal)

# ---------- Nexmon ----------
class SSHAR_Nexmon_Dataset(_SSHARDatasetBase):
    def __init__(self, root_dir, signal='amp', split='train',
                 normalize=True, norm_type='zscore', mean=None, std=None):
        super().__init__(root_dir, device='nexmon', signal=signal, split=split,
                          normalize=normalize, norm_type=norm_type, mean=mean, std=std)

def compute_sshar_nexmon_mean_std(root_dir, signal='amp'):
    return _compute_sshar_mean_std(root_dir, device='nexmon', signal=signal)
```

### 5.4 Cách dùng — 3 dataset, form gọi tương tự nhau

```python
# UT_HAR — normalize minmax theo từng sample (mặc định), hoặc tắt hẳn: normalize=False
train_ut = UTHARDataset(DATA_ROOT, split='train', normalize=True, norm_type='minmax')
test_ut  = UTHARDataset(DATA_ROOT, split='test',  normalize=True, norm_type='minmax')

# SSHAR ESP — zscore (cần mean/std tính từ train trước)
mean, std = compute_sshar_esp_mean_std(DATA_ROOT)
train_esp = SSHAR_ESP_Dataset(DATA_ROOT, split='train', normalize=True, norm_type='zscore', mean=mean, std=std)
test_esp  = SSHAR_ESP_Dataset(DATA_ROOT, split='test',  normalize=True, norm_type='zscore', mean=mean, std=std)

# SSHAR Nexmon — có thể tắt chuẩn hóa để so sánh thử nghiệm
train_nex = SSHAR_Nexmon_Dataset(DATA_ROOT, split='train', normalize=False)
test_nex  = SSHAR_Nexmon_Dataset(DATA_ROOT, split='test',  normalize=False)
```

### 5.5 Nguyên tắc chung áp dụng khi viết lại code loader

- Dùng `os.path.join` xuyên suốt (không hardcode `/`) → tương thích Windows/Kaggle.
- `label` của UT_HAR được giả định có thể ở dạng one-hot (dựa theo cách các dataset kiểu này thường lưu) nên có `argmax` để chuyển về index — **cần người dùng kiểm tra lại file label thực tế**; nếu label vốn đã là index thì bỏ dòng `argmax`.

---

## 6. Câu hỏi còn mở (cần người dùng xác nhận để chốt code)

1. **UT_HAR file split:** đang giả định tên file là `train_data.csv` / `val_data.csv` / `test_data.csv` và tương ứng `train_label.csv`... (theo convention hay gặp ở repo WiFi-CSI-Sensing-Benchmark) — cần confirm đúng tên file thực tế trong `UT_HAR/data/` và `UT_HAR/label/`.
2. **Val set của UT_HAR:** dùng val riêng (để early-stopping / chọn best epoch), hay gộp val vào train luôn và chỉ dùng test để đánh giá — giống cách SSHAR chỉ có train/test?

---

## 7. Tóm tắt các quyết định cuối cùng (phương án đã chốt)

- **Cấu trúc project: dùng bản đơn giản hoá** — 5 file chính: `datasets.py`, `models.py`, `train.py`, `metrics.py`, `main.py` + thư mục `results/` (xem chi tiết mục 2.2).
- **Cấu trúc `results/`: dùng bản đã cập nhật** (xem mục 3.4) — gồm `results/all_history.csv` (tổng hợp toàn bộ 21 tổ hợp) + từng thư mục con `results/{dataset}_{model}/` chứa `history.csv`, `metrics.json`, `confusion_matrix.png`, `learning_curve.png`; đi kèm script vẽ độc lập `plot_compare.py`.
- **Đếm tham số model: dùng `torchinfo`** (giống Keras `model.summary()`) làm phương án chính để lấy bảng summary layer-by-layer và `total_params`/`trainable_params`; fallback bằng `sum(p.numel())` khi `torchinfo` lỗi với kiến trúc đặc biệt (BiMamba). Lưu `total_params`/`trainable_params` vào `metrics.json` (xem mục 4.1).
- **Training:** tối đa 100 epoch, theo dõi metric tốt nhất (test_acc/val_loss) để lưu `best_model.pth`; `best_epoch` được ghi vào `metrics.json`.
- **Path:** dùng `pathlib.Path` / `os.path.join` xuyên suốt, có biến `DATA_ROOT` / `RESULTS_ROOT` để tương thích Windows ↔ Kaggle.
- **Dataset loader:**
  - UT_HAR: giữ nguyên split gốc theo folder train/val/test.
  - SSHAR ESP và SSHAR Nexmon: 2 class riêng biệt (`SSHAR_ESP_Dataset`, `SSHAR_Nexmon_Dataset`), dùng chung logic core bên trong (`_SSHARDatasetBase`).
  - Chuẩn hoá (`normalize`, `norm_type`) là tham số tuỳ chọn cho cả 3 dataset, không hardcode kiểu chuẩn hoá.
  - Input shape lấy tự động từ 1 sample, không hardcode trong `main.py`.

---

*Tài liệu này được tổng hợp từ toàn bộ nội dung trao đổi trong file PDF "Cấu trúc project benchmark mô hình WiFi sensing HAR - Claude".*
