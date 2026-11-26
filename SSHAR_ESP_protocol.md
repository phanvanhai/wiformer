# SSHAR ESP: dữ liệu và quy ước train chung

Note này cố định **thông số thiết bị, cấu trúc dữ liệu, cách chia train/val/test và thiết lập train**
cho dataset SSHAR ESP. Mọi mô hình trong benchmark (AMCS-TCN và các mô hình khác) đều phải theo đúng
note này để kết quả so sánh được với nhau. Code tham chiếu: `datasets/csi_dataset.py`
(`SSHAR_USER_PAIRS`, `sshar_user_folds`, `sshar_user_split_of`, `sshar_split_of`, `_sshar_index`), nhánh `sshar_esp`.

**Mọi cách chia đều 6:2:2** (3840 / 1280 / 1280). Cross-user là 5-fold: fold 0 train 6 người
`01, 03, 04, 09, 11, 12` / val `02, 10` / test `13, 14`. Chi tiết ở mục 3.

Cách gọi trên Kaggle (biến cấu hình, tham số dòng lệnh, tên run, cấu trúc kết quả, cell mẫu):
[kaggle_run_convention.md](kaggle_run_convention.md).

## 1. Thiết bị và tín hiệu

| Mục | Giá trị |
|---|---|
| Thiết bị | ESP32 (thư mục `esp`) |
| Số receiver | 3 (`rx_00`, `rx_01`, `rx_02`) |
| Số anten mỗi receiver | 1 |
| Số subcarrier | 56 |
| Tín hiệu | biên độ CSI (`amp_*.npy`, thư mục `sshar_amp_interp`), float32, giá trị khoảng 1 đến 57 |
| Độ dài mỗi bản ghi | 1000 điểm = 5 s (200 Hz) |
| Shape một file (một receiver) | `(1, 56, 1000)` = (anten, subcarrier, thời gian) |
| Phòng | `room_02` (chỉ có một phòng) |

Một bản ghi là các file cùng tên ở cả 3 receiver, nên một mẫu có shape `(3, 56, 1000)`
(3 receiver × 1 anten = 3 luồng).

## 2. Cấu trúc thư mục và tên file

```
sshar_amp_interp/
  room_02/
    esp/
      rx_00/ subject_01/ amp_act01_pos01_dir00_rep01.npy ...
      rx_01/ subject_01/ ...      # cùng tên file với rx_00
      rx_02/ subject_01/ ...
```

Tên file: `amp_act{AA}_pos{BB}_dir{CC}_rep{DD}.npy`

| Trường | Ý nghĩa | Giá trị |
|---|---|---|
| `act` | hành động, là nhãn | 01 đến 08, ánh xạ sang chỉ số lớp 0 đến 7 (`act - 1`) |
| `pos` | vị trí | 01 đến 08 |
| `dir` | hướng | `00` = không có hướng; `01`, `02` = hai hướng ngược nhau |
| `rep` | lần lặp | 1 đến 10 với dir00, 1 đến 5 với dir01/dir02 |

10 người (subject): `01, 02, 03, 04, 09, 10, 11, 12, 13, 14`.

### Hành động và hướng

Các hành động chia làm hai nhóm. Đây là lý do cách chia tập khác nhau giữa hai nhóm:

| Hành động | Hướng | Số lần lặp | Số bản ghi mỗi người, mỗi hành động |
|---|---|---|---|
| act01, act02, act03, act08 | chỉ `dir00` | 10 | 8 vị trí × 10 lần = **80** |
| act04, act05, act06, act07 | `dir01` và `dir02` | 5 mỗi hướng | 8 vị trí × 2 hướng × 5 lần = **80** |

Tổng: 10 người × 8 hành động × 80 = **6400 bản ghi** (800 mỗi lớp, cân bằng).
Tên hành động: act01 ngồi xuống, act02 đứng lên, act03 cúi nhặt, act04 ngã, act05 nằm im, act06 đi, act07 chạy, act08 phòng trống.

### Số lớp khi train (`--labels`)

| `--labels` | Lớp (chỉ số) | Mặc định cho |
|---|---|---|
| 8 | ngồi xuống 0, đứng lên 1, cúi nhặt 2, ngã 3, nằm im 4, đi 5, chạy 6, phòng trống 7 | các mô hình khác |
| **7** | ngồi xuống 0, đứng lên 1, cúi nhặt 2, ngã 3, đi 4, chạy 5, **tĩnh 6** (= nằm im + phòng trống) | `wiformer_v5` |

Với 7 lớp, nằm im và phòng trống được gộp thành một lớp "tĩnh", đóng vai trò lớp **không có hành động**.
Sau khi trừ mức trung bình theo thời gian của từng kênh, hai trạng thái này chỉ còn khác nhau ở nhiễu.
Lớp tĩnh có gấp đôi số mẫu của mỗi lớp hành động (1600 so với 800). Mức mất cân bằng 2:1 này nhẹ nên giữ
nguyên, không đặt trọng số lớp. Thước đo chính là độ chính xác trên 6 hành động; 7 lớp là thước đo phụ.

## 3. Chia train / val / test (6:2:2)

Mọi cách chia đều theo tỉ lệ **6:2:2**, giống benchmark `CodeForPaper/benchmark`
(`doc/experiment_plan_v2.md`). Cả hai cách chia chính đều ra **3840 / 1280 / 1280** mẫu
(mỗi lớp 480 / 160 / 160). Trong mọi trường hợp, một bản ghi (cả 3 receiver) luôn nằm trọn trong một
tập, và không xáo trộn ngẫu nhiên từng mẫu rồi mới chia.

Protocol cũ 7:1:2 (4480 / 640 / 1280) đã bỏ khỏi code.

### 3A. Cross-user (`--split_mode user`)

Chia **theo người**: mỗi người chỉ thuộc đúng một tập, nên tập test gồm những người model chưa từng
thấy khi train. Mỗi người có 640 bản ghi (8 hành động × 80), nên 6 / 2 / 2 người tương ứng
3840 / 1280 / 1280 mẫu.

**5-fold theo người.** 10 người được ghép thành 5 cặp cố định (`SSHAR_USER_PAIRS`). Fold f: test = cặp f,
val = cặp (f + 1) mod 5, train = 6 người còn lại. Mỗi người được **test đúng một lần** và làm val đúng
một lần. Báo cáo trung bình ± độ lệch chuẩn qua 5 fold.

| Fold | Train (6 người) | Val (2 người) | Test (2 người) |
|---|---|---|---|
| **0** | **01, 03, 04, 09, 11, 12** | **02, 10** | **13, 14** |
| 1 | 01, 04, 09, 12, 13, 14 | 03, 11 | 02, 10 |
| 2 | 02, 04, 10, 12, 13, 14 | 01, 09 | 03, 11 |
| 3 | 02, 03, 10, 11, 13, 14 | 04, 12 | 01, 09 |
| 4 | 01, 02, 03, 09, 10, 11 | 13, 14 | 04, 12 |

Bộ 5 case cũ (người 03 bị test 2 lần, người 04 không bao giờ được test) đã bỏ. Kết quả cross-user cũ
không so sánh trực tiếp được với các fold mới, trừ tập test của fold 0 (vẫn là 13, 14).

```python
SSHAR_ALL_SUBJECTS = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14]
SSHAR_USER_PAIRS = [(13, 14), (2, 10), (3, 11), (1, 9), (4, 12)]

def sshar_user_folds(user_case=0):
    test = list(SSHAR_USER_PAIRS[user_case])
    val = list(SSHAR_USER_PAIRS[(user_case + 1) % 5])
    train = [s for s in SSHAR_ALL_SUBJECTS if s not in test + val]
    return train, val, test

def sshar_user_split_of(subject_id, user_case=0):
    _, val, test = sshar_user_folds(user_case)
    return 'test' if subject_id in test else 'val' if subject_id in val else 'train'
```

### 3B. Chia theo lần lặp (`--split_mode repeat`)

Chia **theo lần lặp**: người nào, vị trí nào cũng có mặt ở cả ba tập. Với **mỗi cặp (người, hành
động)**, 80 bản ghi được chia đúng **48 / 16 / 16**. Quy tắc giống hệt `v2_split_of` trong benchmark.

| Tập | `dir00` (act01, 02, 03, 08), 10 lần lặp | `dir01` / `dir02` (act04, 05, 06, 07), 5 lần lặp mỗi hướng |
|---|---|---|
| train | rep 1 đến 6 (8 vị trí × 6 = 48) | rep 1 đến 3 của cả hai hướng (8 × 2 × 3 = 48) |
| val | rep 7 và 8 (8 × 2 = 16) | rep 4 của cả hai hướng (8 × 2 = 16) |
| test | rep 9 và 10 (8 × 2 = 16) | rep 5 của cả hai hướng (8 × 2 = 16) |

```python
def sshar_split_of(pos, direction, rep):
    if direction == 0:                      # 10 lần lặp
        return 'test' if rep > 8 else 'val' if rep > 6 else 'train'
    return 'test' if rep == 5 else 'val' if rep == 4 else 'train'   # 5 lần lặp mỗi hướng
```

Tập test giống hệt protocol cũ 7:1:2; chỉ phần train/val khác.

### 3C. Cross-position (`--split_mode position`, tuỳ chọn)

Test là 2/8 vị trí (`pos % 4 == case`, case 0..3). Phần còn lại chia ngẫu nhiên train:val = **3:1**
(= 6:2), phân tầng theo lớp, seed 42. Vì có 8 vị trí nên test luôn là 25%, không đúng 20%.

### Số mẫu phải ra (repeat và user)

| Tập | Tổng | Mỗi lớp |
|---|---|---|
| train | 3840 | 480 |
| val | 1280 | 160 |
| test | 1280 | 160 |

## 4. Thiết lập train (chung cho mọi mô hình)

| Thiết lập | Giá trị |
|---|---|
| Batch size | 32 |
| Epochs | tối đa 100 |
| Optimizer | Adam |
| Learning rate | theo bài báo gốc của từng mô hình (AMCS-TCN: 0.001, Table 1 của bài) |
| Chọn model | checkpoint có **val accuracy** cao nhất |
| Dừng sớm (early stopping) | patience **10 epoch** theo val accuracy (xem bên dưới) |
| Metric | accuracy, precision, recall, F1 (precision/recall/F1 lấy trung bình **macro** trên các lớp train, 8 hoặc 7, xem `--labels`) |
| Seed | hiện chỉ dùng 42 để đánh giá nhanh, sau này có thể thêm seed |

**Dừng sớm**, cách tính cụ thể để mọi mô hình làm giống nhau:

- Sau mỗi epoch, đo accuracy trên val. Nếu val accuracy **lớn hơn hoặc bằng** mức tốt nhất trước đó,
  lưu checkpoint mới và đặt bộ đếm về 0. Ngược lại, bộ đếm tăng thêm 1.
- Khi bộ đếm đạt **10** (10 epoch liên tiếp không cải thiện), dừng train.
- Checkpoint dùng để đánh giá là checkpoint đã lưu cuối cùng, tức là có val accuracy cao nhất. Nếu
  nhiều epoch hoà nhau thì lấy epoch muộn nhất.
- Số epoch tối đa vẫn là 100. Báo cáo kèm best epoch và tổng số epoch đã chạy.

**Luồng mỗi lần chạy** (không chia chế độ dev/final): train trên train, chọn checkpoint theo val (val
chỉ dùng để chọn checkpoint và dừng sớm), rồi báo cáo trên **test** bằng checkpoint đó, kèm kết quả val
của cùng checkpoint để đối chiếu. Hiện chạy seed 42; khi có thêm seed thì báo cáo mean ± std qua các seed.

Những mục thuộc về từng mô hình (ghi rõ trong báo cáo của mô hình đó, không phải thiết lập chung):
LR scheduler, augmentation, class weights, chuẩn hoá input và cách reshape input.
Để tham khảo, AMCS-TCN dùng StepLR(20, 0.5), nhiễu Gaussian
std 0.02, min-max theo min/max toàn cục của tập train, và giữ đủ T = 1000 với input `(3, 1000, 56)`.

## 5. Checklist khi thêm một mô hình mới

- [ ] Đọc đủ 3 receiver, và bản ghi nào cũng có đủ ở cả ba
- [ ] Nhãn `act01..08` ánh xạ sang `0..7`
- [ ] Chia 6:2:2 theo mục 3 (cross-user fold 0: train 01, 03, 04, 09, 11, 12 / val 02, 10 / test 13, 14,
      hoặc theo lần lặp); số mẫu khớp **3840 / 1280 / 1280** (mỗi lớp 480 / 160 / 160)
- [ ] Ghi rõ cách chia trong báo cáo (cross-user và case số mấy, hoặc theo lần lặp)
- [ ] Thống kê chuẩn hoá (nếu có) chỉ tính trên **train**
- [ ] Batch 32, tối đa 100 epoch, Adam, LR theo bài gốc
- [ ] Chọn checkpoint theo val accuracy cao nhất
- [ ] Dừng sớm sau 10 epoch liên tiếp val accuracy không cải thiện (bằng mức tốt nhất vẫn tính là cải thiện)
- [ ] Báo cáo trên test bằng checkpoint chọn theo val, kèm kết quả val của cùng checkpoint (seed 42)
- [ ] Báo cáo accuracy và precision / recall / F1 (macro)
