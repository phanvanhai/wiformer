# WiFormer v4 — Tổng kết kiến trúc và kết quả

Tài liệu tổng kết: tư tưởng thiết kế từng khối, kết quả trên SSHAR-ESP (và UT-HAR), và đối chiếu kết quả với các giả thuyết đặt ra khi thiết kế. Chi tiết shape và sơ đồ: [WiFormer_v4_Architecture.md](WiFormer_v4_Architecture.md). Code: [models/wiformer_v4.py](models/wiformer_v4.py).

Trong tài liệu này, các phiên bản cũ được coi là ablation của v4:
- **`abs_pos`** = v2 với chuẩn hóa của v4: chỉ khác v4 ở **cơ chế mã hóa vị trí** của Khối 2–3.
- **`late_fusion`** = v3: chỉ khác v4 ở **cách gộp hai góc nhìn** (gộp muộn hai luồng, có loss phụ; kèm cửa sổ 60 thay vì 64).

---

## 1. Tư tưởng chung

CSI biên độ của một mẫu là một khối 3 chiều: **RX × subcarrier × thời gian**. Mỗi chiều mang một loại thông tin khác nhau, nên mô hình xử lý lần lượt từng thang:

| Thang | Thông tin | Khối xử lý |
|---|---|---|
| Trong một đoạn ngắn (0.32 s) của một kênh (rx, sub) | Hình dạng dao động, tốc độ thay đổi (Doppler) | Khối 1 |
| Giữa các kênh trong cùng một đoạn | Kênh nào nhạy với cử động, mẫu không gian của nhiễu xạ | Khối 2 |
| Giữa các đoạn theo thời gian | Diễn biến của hành động: bắt đầu, kết thúc, thứ tự pha | Khối 3 |

Ba nhận xét định hướng thiết kế:
1. **Mỗi đoạn có hai góc nhìn bổ sung nhau.** Dạng sóng (thời gian) cho biết hình dạng cử động. Phổ (tần số) cho biết tốc độ, nhịp. Hai góc nhìn này khớp nhau đến từng ô (rx, sub, đoạn), nên nên gộp **ngay tại token** (gộp sớm), thay vì xử lý hai luồng rồi gộp ở logit (gộp muộn).
2. **Vị trí tương đối quan trọng hơn vị trí tuyệt đối.** Subcarrier gần nhau thì tương quan mạnh, bất kể nằm ở đâu trong dải. Thứ tự trước/sau giữa các đoạn quyết định hành động (ngồi xuống và đứng lên gần như là ảnh gương theo thời gian).
3. **Mức tĩnh của tín hiệu là dấu vân tay của môi trường và người.** Bỏ mức tĩnh giúp tổng quát sang người mới, nhưng làm khó phân biệt hai trạng thái tĩnh (nằm im ↔ phòng trống).

## 2. Kiến trúc từng khối

### 2.0 Tiền xử lý: chuẩn hóa theo từng mẫu

- **Làm gì:** với mỗi mẫu, trừ mean theo thời gian của từng (rx, sub), rồi chia **một std chung cho mỗi rx**. Không dùng z-score toàn cục. Chỉ dùng anten 0 của mỗi RX.
- **Vì sao:**
  - Trừ mean từng kênh: bỏ mức tĩnh (dấu vân tay môi trường/người). Ở v2, chuẩn hóa từng mẫu cho user 0 đạt 90.86% so với 72.11% khi dùng z-score toàn cục.
  - Chia std theo rx thay vì theo từng kênh: **giữ độ mạnh tương đối giữa các subcarrier**. Subcarrier nhạy với cử động có biên độ dao động lớn hơn; chia theo từng kênh sẽ san phẳng điều này.
- **Cái giá:** mất thông tin mức tĩnh, nên nằm im ↔ phòng trống khó hơn.
- **UT-HAR:** dùng `norm=none` (giữ min-max của loader), vì trên UT-HAR mức tĩnh là thông tin then chốt.

### 2.1 Khối 1: hai góc nhìn của một đoạn, gộp sớm

- **Làm gì:**
  - Cắt mỗi kênh (rx, sub) thành các đoạn W = 64 mẫu (0.32 s ở 200 Hz), bước 32 (chồng 50%). Trên ESP được K = 30 đoạn.
  - **Góc nhìn thời gian:** 3 lớp Conv1d (1→32→64→64), pool `flat4` (giữ 4 bước thời gian trong đoạn), `Linear(256 → 64)`.
  - **Góc nhìn phổ:** `rfft` → `log1p(|X|)` → bỏ DC → 32 bin (3.125–100 Hz) → `Linear(32 → 64)`. Tương đương STFT cửa sổ chữ nhật.
  - **Gộp:** `Linear(128 → 64)` trên `[v_time ‖ v_freq]`, cho ra một token cho mỗi (rx, sub, đoạn). Trọng số dùng chung cho mọi kênh.
- **Vì sao:**
  - Ở v3, gộp muộn hai luồng làm trọng số gộp theo lớp kẹt ở 0.43–0.59 và luồng phổ bị yếu đi khi train chung (*modality imbalance*). Hai góc nhìn khớp nhau từng ô thì gộp sớm tự nhiên hơn, và cho phép tương tác thời gian × tần số ở mọi tầng sau.
  - W là lũy thừa của 2 để Conv (hai lớp stride 2) và pool `flat4` chia đều.

### 2.2 Khối 2: attention giữa RX × subcarrier trong từng đoạn

- **Làm gì:** với mỗi đoạn, attention trên 3 × 56 = 168 token + 1 CLS, 2 lớp, 4 head. Bias attention:
  - cùng RX: `b_sub[|Δs|]` theo **khoảng cách subcarrier**;
  - khác RX: một scalar `c_cross`;
  - bias riêng cho CLS (CLS tách khỏi vị trí).

  Đầu ra là CLS của từng đoạn, cho chuỗi K = 30 token.
- **Vì sao:**
  - Không phải subcarrier nào cũng nhạy như nhau. Lấy trung bình làm loãng kênh nhạy, còn attention tự chọn kênh.
  - Bias theo khoảng cách subcarrier phản ánh tương quan cục bộ trong dải tần.
  - Không gán danh tính cho RX, nên mô hình bất biến khi hoán vị RX (kiểm tra: 100% dự đoán giữ nguyên).

### 2.3 Khối 3: Transformer thời gian giữa các đoạn

- **Làm gì:** attention trên 30 token đoạn + CLS, 6 lớp, 4 head. Bias **có dấu, học được** `b[i − j]` (kiểu T5/Swin), CLS tách vị trí. CLS cuối qua classifier (LayerNorm, Dropout, `Linear(64 → C)`). Loss chỉ là CE.
- **Vì sao:**
  - Bias có dấu phân biệt "trước" với "sau". ALiBi (v2) đối xứng, chỉ biết khoảng cách, nên không tách được hai hành động ngược chiều thời gian như ngồi xuống ↔ đứng lên.
  - Phổ trong một đoạn không có chiều thời gian, nhưng chuỗi đoạn qua Khối 3 thì có. Nhịp điệu bất đối xứng vẫn được giữ.

### 2.4 Quy mô

| | Params | MACs (ESP) |
|---|---|---|
| v4 full | 449K | 2.41 G |

Gần như toàn bộ MACs nằm ở Conv của Khối 1. Góc nhìn phổ chỉ tốn khoảng 6K tham số và 31M MACs.

## 3. Thiết lập thí nghiệm

- **Dữ liệu:** SSHAR-ESP, phòng 2, 3 RX × 56 subcarrier, 8 lớp: ngồi xuống, đứng lên, cúi nhặt, ngã, nằm im, đi, chạy, phòng trống.
- **Split:** repeat 6:2:2 theo lần lặp (3840 / 1280 / 1280), giống benchmark CNN-GRU và BiMamba (cùng tập test, cùng thứ tự nhãn).
- **Huấn luyện:** Adam, lr 1e-3, batch 32, warmup 5 epoch, cosine về 0 ở epoch 100, không dừng sớm, checkpoint theo val acc cao nhất, seed 42. Mỗi cấu hình 1 lần chạy.
- **Ba mức đánh giá** (tính từ cùng một ma trận nhầm lẫn, không train lại):
  - **8 nhãn:** đầy đủ.
  - **7 nhãn:** gộp nằm im và phòng trống thành "tĩnh".
  - **6 nhãn:** chỉ các mẫu hành động (960 mẫu).
- **Baseline:** CNN-GRU và BiMamba dùng z-score toàn cục (thống kê tập train), lr 1e-3 cố định, batch 64.

## 4. Kết quả

### 4.1 So với baseline (ESP repeat)

| Model | Acc 8 | F1 8 | Acc 7 | F1 7 | Acc 6 | F1 6 | Params | MACs |
|---|---|---|---|---|---|---|---|---|
| **WiFormer v4** | **96.80** | **96.79** | **97.97** | **97.68** | **97.29** | **97.29** | 449K | 2.41 G |
| BiMamba | 95.31 | 95.31 | 95.31 | 94.68 | 93.85 | 93.95 | 653K | 0.10 G |
| CNN-GRU | 94.92 | 94.91 | 94.92 | 94.20 | 93.23 | 93.31 | 2.0M | 6.70 G |

### 4.2 Ablation (ESP repeat)

| Biến thể | Thay đổi so với full | Acc 8 | Δ8 | Acc 7 | Δ7 | Acc 6 | Δ6 | MACs |
|---|---|---|---|---|---|---|---|---|
| **full** | — | **96.80** | — | **97.97** | — | **97.29** | — | 2.41 G |
| *Khối 1* | | | | | | | | |
| wo_fft | chỉ góc nhìn thời gian | 95.86 | −0.94 | 97.19 | −0.78 | 96.25 | −1.04 | 2.38 G |
| wo_conv | chỉ góc nhìn phổ | 95.16 | −1.64 | 97.27 | −0.70 | 96.35 | −0.94 | 0.54 G |
| late_fusion (= v3) | gộp muộn hai luồng + loss phụ | 96.09 | −0.71 | 97.34 | −0.63 | 96.46 | −0.83 | 2.40 G |
| wo_block1 | `Linear(W → D)` trên đoạn thô | 91.72 | **−5.08** | 96.02 | **−1.95** | 94.69 | **−2.60** | 0.53 G |
| *Mã hóa vị trí* | | | | | | | | |
| abs_pos (= v2) | Khối 2: embedding subcarrier tuyệt đối; Khối 3: ALiBi | 95.62 | −1.18 | 96.64 | −1.33 | 95.52 | −1.77 | 2.24 G |
| *Khối 2, Khối 3* | | | | | | | | |
| wo_block2 | trung bình 168 token mỗi đoạn | 94.61 | −2.19 | 97.42 | −0.55 | 96.56 | −0.73 | 1.91 G |
| wo_block3 | trung bình các đoạn | *(bổ sung)* | | | | | | 2.40 G |

### 4.3 Lỗi theo cặp lớp (số mẫu test bị nhầm, cả hai chiều)

| | ngồi ↔ đứng | nội bộ ngồi xuống / đứng lên / cúi nhặt | đi ↔ chạy | nằm im ↔ phòng trống | Tổng lỗi |
|---|---|---|---|---|---|
| **full** | 6 | **8** | 16 | 15 | **41** |
| wo_fft | 10 | 12 | 22 | 17 | 53 |
| wo_conv | 13 | 17 | 16 | 27 | 62 |
| late_fusion (= v3) | **5** | **8** | 24 | 16 | 50 |
| wo_block1 | 14 | 19 | 24 | 55 | 106 |
| abs_pos (= v2) | 12 | 21 | 20 | 13 | 56 |
| wo_block2 | 11 | 14 | 18 | 36 | 69 |
| wo_block3 | | | | | |
| v3 chỉ luồng phổ (`wo_main`) | 14 | 18 | **10** | 20 | 49 |
| BiMamba | 11 | 13 | 37 | **0** | 60 |
| CNN-GRU | 12 | 13 | 39 | **0** | 65 |

### 4.4 Mô hình dựa vào góc nhìn nào (knock-out lúc test trên v4 full)

Zero một nửa đầu vào của lớp gộp lúc test, không train lại. Chuẩn trọng số gộp: thời gian 7.05, phổ 7.66.

| Recall (%) | ngồi xuống | đứng lên | cúi nhặt | ngã | nằm im | đi | chạy | phòng trống | Acc |
|---|---|---|---|---|---|---|---|---|---|
| Cả hai | 96.9 | 98.1 | 98.8 | 100 | 91.9 | 98.1 | 91.9 | 98.8 | 96.8 |
| Chỉ phổ (zero thời gian) | 60.6 | 100 | 83.8 | 57.5 | **0** | 98.8 | 68.8 | 100 | 71.2 |
| Chỉ thời gian (zero phổ) | 45.0 | 75.6 | 70.0 | 93.1 | 43.1 | **48.8** | 97.5 | **28.8** | 62.7 |

### 4.5 UT-HAR (split chính thức)

| | Acc | F1 |
|---|---|---|
| AMCS-TCN (5 seed) | 97.36 ± 0.34 | 95.88 |
| late_fusion (= v3), `norm=none` | 94.8 | 91.9 |
| WiFormer v4, `norm=none` | 93.4 | 90.6 |
| WiFormer v4, chuẩn hóa từng mẫu | 71.2 | 58.8 |

## 5. Đối chiếu ý tưởng với kết quả

| Ý tưởng / dự đoán khi thiết kế | Bằng chứng | Kết luận |
|---|---|---|
| **Khối 1:** trích đặc trưng trong đoạn là cần thiết, phép chiếu thô không đủ | wo_block1 tụt mạnh nhất (−5.08 / −2.60 ở 6 nhãn), overfit rõ (val loss thấp nhất 0.271 so với 0.089) | **Đúng** |
| **Hai góc nhìn bổ sung nhau:** phổ cho tốc độ (đi ↔ chạy), thời gian cho hình dạng (chuyển tư thế, trạng thái tĩnh) | Bỏ phổ: đi ↔ chạy 16 → 22. Bỏ thời gian: nằm im ↔ phòng trống 15 → 27, nội bộ ngồi/đứng/cúi 8 → 17. Knock-out: chỉ phổ thì nằm im = 0; chỉ thời gian thì đi = 48.8. Gộp hai góc nhìn tốt hơn từng góc nhìn ở cả 3 mức | **Đúng** |
| **Gộp sớm tốt hơn gộp muộn,** không còn modality imbalance | full hơn late_fusion 0.7–0.8 điểm ở cả 3 mức; đi ↔ chạy 24 → 16; các cặp khác giữ nguyên. Trọng số gộp hai nửa gần bằng nhau (7.05 / 7.66), zero bên nào cũng mất 25–34 điểm | **Đúng** |
| Gộp sớm cho đi ↔ chạy gần bằng luồng phổ riêng của v3 (10 lỗi) | full còn 16 lỗi, giữa v3 full (24) và v3 `wo_main` (10) | **Đúng một phần** |
| Phổ từng subcarrier qua Khối 2 tốt hơn phổ lấy trung bình (`wo_conv` ≥ v3 `wo_main`) | wo_conv 95.16 < v3 `wo_main` 96.17; đi ↔ chạy 16 so với 10 | **Sai.** Luồng phổ của v3 (115K tham số, 4.1M MACs) mạnh hơn khi đứng một mình. Cần xem lại cách biểu diễn phổ trong token |
| **Mã hóa vị trí tương đối, có dấu** giúp các hành động ngược chiều thời gian | abs_pos: ngồi ↔ đứng 6 → 12, nội bộ ngồi/đứng/cúi 8 → 21. Mức tụt tăng dần khi bỏ cặp tĩnh (−1.18 → −1.33 → −1.77), tức lợi ích nằm ở chính các hành động | **Đúng** |
| **Khối 2:** attention chọn kênh nhạy tốt hơn trung bình | wo_block2 −2.19 ở 8 nhãn, nhưng chỉ −0.55 / −0.73 ở 7 / 6 nhãn. Lỗi tăng chủ yếu ở nằm im ↔ phòng trống (15 → 36) | **Đúng, nhưng ở split repeat vai trò chính là phân biệt trạng thái tĩnh** (vết thở/cử động nhỏ chỉ ở vài kênh). Vai trò cho cross-user chưa kiểm tra (v2 ở user 0: bỏ Khối 2 tụt 11.8 điểm) |
| **Khối 3:** diễn biến thời gian giữa các đoạn là cần thiết | | *Chờ wo_block3* |
| **Chuẩn hóa từng mẫu:** đổi khả năng phân biệt hai trạng thái tĩnh lấy khả năng tổng quát | v4 thua baseline duy nhất ở nằm im ↔ phòng trống (15 so với 0). Trên 6 hành động, v4 hơn baseline 3.4–4.1 điểm. UT-HAR cần giữ mức tĩnh (93.4 so với 71.2). Phần "tổng quát sang người mới" mới có bằng chứng từ v2 | **Đúng về cái giá; phần lợi ích cross-user chưa kiểm tra trên v4** |

## 6. Nhận xét về mô hình

1. **WiFormer v4 tốt nhất trong mọi cấu hình đã chạy** ở cả 8, 7 và 6 nhãn. Hơn baseline 1.5–1.9 điểm ở 8 nhãn và 3.4–4.1 điểm khi chỉ xét hành động.
2. **Mọi thành phần đều có đóng góp dương.** Thứ tự tác động trên các hành động (6 nhãn): Khối 1 (−2.60) > mã hóa vị trí (−1.77) > góc nhìn thời gian (−1.04) > góc nhìn phổ (−0.94) > gộp muộn (−0.83) > Khối 2 (−0.73).
3. **Mỗi thành phần phục vụ một nhóm lỗi rõ ràng,** khớp với lý do thiết kế:

   | Nhóm lỗi | Thành phần chịu trách nhiệm |
   |---|---|
   | đi ↔ chạy | góc nhìn phổ, gộp sớm |
   | ngồi xuống ↔ đứng lên | mã hóa vị trí có dấu, góc nhìn thời gian |
   | nằm im ↔ phòng trống | góc nhìn thời gian, Khối 2 |

4. **Điểm yếu còn lại:**
   - **Nằm im ↔ phòng trống (15 lỗi)** là cái giá của chuẩn hóa từng mẫu. Hướng xử lý có nguyên tắc là khai thác vi chuyển động (nhịp thở 0.2–0.5 Hz). Độ phân giải phổ hiện tại (3.125 Hz trong đoạn 0.32 s) không thấy được nhịp này, chỉ Khối 3 có thể học qua biến thiên giữa các đoạn.
   - **Chạy → đi (13 mẫu).**
   - **Góc nhìn phổ đứng một mình yếu hơn luồng phổ của v3.**
   - **UT-HAR:** kém AMCS-TCN khoảng 4 điểm.
5. **Chi phí:** góc nhìn phổ rất rẻ (`wo_conv` 0.54 G MACs, 96.35% ở 6 nhãn). Nếu cần một bản nhẹ, có thể dùng chỉ góc nhìn phổ.

## 7. Hạn chế và việc còn lại

1. **wo_block3** (bổ sung).
2. **Cross-user:** mọi kết luận trên đều ở split repeat, nơi val gần bão hòa. Cần đánh giá theo người, tốt nhất là 5 fold rời nhau theo người. Lưu ý: 5 case user hiện tại chưa phải 5 fold đúng nghĩa (người 3 được test hai lần, người 4 không bao giờ được test).
3. **Một seed:** các chênh lệch khoảng 0.7–1 điểm (khoảng 9–12 mẫu) cần thêm seed để khẳng định.
4. **Ablation chưa sạch:**
   - `late_fusion` lẫn thêm cửa sổ 60 và loss phụ.
   - `abs_pos` đổi cả Khối 2 và Khối 3 cùng lúc; chưa tách được đóng góp của từng khối.
