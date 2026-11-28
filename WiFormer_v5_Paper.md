# WiFormer v5: nhận dạng hoạt động bằng CSI WiFi như một chuỗi khoảnh khắc, mỗi khoảnh khắc có hình dạng và tốc độ

*Bản thảo nội bộ, cập nhật 2026-11-28. Dữ liệu: SSHAR-ESP (room_02), đánh giá chéo người (cross-user) 5 fold, mỗi cấu hình 1 seed. Một số ablation mới có 2–3/5 fold. Các ô "(bổ sung)" sẽ điền khi có kết quả.*

---

## Tóm tắt

Biên độ CSI WiFi của một hoạt động thay đổi theo ba trục: theo thời gian, giữa các subcarrier và giữa các bộ thu (RX). WiFormer v5 coi một hoạt động là **một chuỗi các khoảnh khắc ngắn** (khoảng 0.3 s). Mỗi khoảnh khắc được mô tả bằng hai thông tin bổ sung cho nhau:

- **Hình dạng chuyển động:** mỗi kênh (RX, subcarrier) thấy chuyển động theo một cách khác nhau. Mô hình nhìn từng kênh bằng Conv1d, rồi gộp có chọn lọc bằng attention RX × subcarrier.
- **Tốc độ chuyển động:** gần như giống nhau trên mọi kênh. Mô hình lấy phổ biên độ (cửa sổ Hann) của từng kênh rồi trung bình qua mọi kênh.

Hai mô tả được gộp tuyến tính theo từng khoảnh khắc, có modality dropout trên nhánh hình dạng. Sau đó một Transformer thời gian với bias tương đối **có dấu** đọc chuỗi khoảnh khắc theo thứ tự.

Trên SSHAR-ESP (10 người, 3 RX ESP32, 7 lớp), với đánh giá chéo người 5 fold, mô hình đạt:
- **92.90 ± 1.78 %** accuracy trên 6 hành động;
- 94.66 ± 1.30 % accuracy 7 lớp;
- 93.81 % macro-F1 7 lớp.

Mô hình có 453K tham số và khoảng 2.4 GMACs mỗi mẫu.

**Ablation:** mọi thành phần đều có đóng góp có ý nghĩa thống kê (McNemar gộp, p < 0.004).
- Thứ tự giữa các khoảnh khắc và việc chia chuỗi thành khoảnh khắc quan trọng nhất: bỏ đi thì mất 6–8 điểm.
- Các thành phần còn lại đóng góp 1.4–3.9 điểm.
- Mỗi thành phần sửa đúng nhóm lỗi mà nó được thiết kế để sửa, và hầu hết lợi ích nằm ở các hành động tại chỗ có cùng tư thế (ngồi / đứng / ngã).

**Phân tích lỗi:** khoảng một nửa lỗi còn lại là "chạy → đi" của ba người. Phổ tốc độ cho thấy những người này chạy gần như với tốc độ đi bộ, nên đây là giới hạn của dữ liệu và của việc tổng quát hóa sang người mới, không phải của kiến trúc.

---

## 1. Giới thiệu

Nhận dạng hoạt động bằng WiFi (WiFi sensing) dùng thông tin trạng thái kênh (CSI) mà bộ thu đo được trên từng subcarrier. Khi người chuyển động, các đường phản xạ thay đổi, nên biên độ CSI dao động. Tín hiệu này có ba đặc điểm khiến bài toán khó:

1. **Mỗi kênh nhìn một phiên bản khác của cùng một chuyển động.** Biên độ và pha của dao động phụ thuộc vào vị trí RX và tần số subcarrier. Một số kênh rất nhạy, một số gần như không thấy gì.
2. **Tốc độ chuyển động lại là đại lượng chung.** Tần số Doppler do cơ thể gây ra gần như giống nhau trên mọi kênh, chỉ khác về cường độ.
3. **Hành động là một diễn biến theo thời gian.** Nhiều cặp hành động có cùng các tư thế nhưng khác thứ tự, ví dụ ngồi xuống và đứng lên, hoặc khác tốc độ, ví dụ ngồi xuống và ngã.

Các mô hình học sâu phổ biến cho CSI thường coi dữ liệu là ảnh hoặc chuỗi đa kênh, và xử lý mọi kênh theo cùng một cách. WiFormer v5 đi theo hướng ngược lại: tách thông tin **riêng của từng kênh** (hình dạng) khỏi thông tin **chung của mọi kênh** (tốc độ), và xử lý mỗi loại theo cách phù hợp với bản chất của nó.

**Đóng góp:**

1. **Biểu diễn "khoảnh khắc hai nhánh".** Mỗi khoảnh khắc ngắn có hai nhánh:
   - nhánh hình dạng: Conv1d trên từng kênh, rồi attention RX × subcarrier với bias tương đối, bất biến khi hoán vị RX;
   - nhánh tốc độ: phổ Hann trung bình qua mọi kênh.

   Hai nhánh gộp tuyến tính theo từng khoảnh khắc.
2. **Modality dropout để cân bằng hai nhánh** có kích thước chênh nhau khoảng 70 lần. So sánh với cách gộp bằng cổng vô hướng cho thấy cổng bị "sập" về nhánh nhỏ hơn, còn gộp tuyến tính kèm dropout dùng được cả hai nhánh.
3. **Transformer thời gian với bias tương đối có dấu.** Bias học được nghiêng hẳn về một hướng thời gian, nhất quán trên cả 5 fold. Bỏ thông tin hướng làm tăng gần gấp đôi lỗi ở các hành động tại chỗ.
4. **Đánh giá chéo người nghiêm ngặt và phân tích lỗi chi tiết.** Đánh giá gồm 5 fold, McNemar gộp, và phân tích theo nhóm hành động, theo người, theo phổ tốc độ. Phân tích chỉ ra rằng lỗi chính còn lại là do khác biệt giữa người.

## 2. Các nghiên cứu liên quan

**Nhận dạng hoạt động bằng CSI WiFi.**
- CARM (Wang et al., MobiCom 2015) liên hệ tốc độ thay đổi của CSI với tốc độ chuyển động thông qua phân tích thời gian–tần số.
- Widar3.0 (Zheng et al., MobiSys 2019) dùng phổ Doppler và đặc trưng độc lập với vị trí để tổng quát sang miền mới.
- Các mô hình học sâu (CNN, LSTM, CNN-GRU, ViT) được tổng hợp trong SenseFi (Yang et al., Patterns 2023), cùng bộ dữ liệu UT-HAR (Yousefi et al., IEEE Commun. Mag. 2017).
- Đa số các mô hình này chọn **một** miền biểu diễn (chuỗi biên độ hoặc spectrogram) và không tách thông tin riêng của từng kênh khỏi thông tin chung.

**Transformer cho CSI.**
- THAT (Li et al., AAAI 2021) dùng hai luồng Transformer theo thời gian và theo kênh, rồi gộp ở cuối.
- Các biến thể ViT chia CSI thành patch 2D, thường với embedding vị trí tuyệt đối và không tách riêng chiều RX.

**Mã hóa vị trí tương đối.**
- Shaw et al. (NAACL 2018) đưa vị trí tương đối vào attention. T5 (Raffel et al., JMLR 2020) và Swin (Liu et al., ICCV 2021) dùng bias vô hướng học được theo khoảng cách.
- ALiBi (Press et al., ICLR 2022) dùng bias tuyến tính cố định theo `|i − j|`, đối xứng hai chiều.
- WiFormer v5 dùng bias **có dấu** theo thời gian, và bias theo khoảng cách subcarrier kèm một bias chung cho cặp khác RX trong không gian.

**Cửa sổ phổ.** Cửa sổ Hann giảm rò rỉ phổ so với cửa sổ chữ nhật (Harris, Proc. IEEE 1978). Với overlap 50 %, tổng các cửa sổ Hann tuần hoàn là hằng số (COLA), như trong STFT.

**Gộp đa nguồn và modality dropout.**
- ModDrop (Neverova et al., TPAMI 2016) bỏ ngẫu nhiên cả một nguồn khi train để mô hình không phụ thuộc vào một nguồn.
- Wang, Tran & Feiszli (CVPR 2020) chỉ ra rằng mạng đa nguồn có xu hướng học "tham lam" theo nguồn dễ hơn.
- Gated Multimodal Unit (Arevalo et al., 2017) gộp hai nguồn bằng cổng học được.

Mục 5.6 cho thấy cơ chế cổng gặp đúng hiện tượng học tham lam nói trên.

## 3. Phương pháp

### 3.1 Ký hiệu và tổng quan

| Ký hiệu | Ý nghĩa | SSHAR-ESP |
|---|---|---|
| R | số RX | 3 |
| S | số subcarrier | 56 |
| T, fs | độ dài chuỗi, tần số lấy mẫu | 1000 mẫu, 200 Hz (5 s) |
| W, stride | độ dài khoảnh khắc, bước nhảy | 64 mẫu (0.32 s), 32 |
| K | số khoảnh khắc | 30 |
| D | kích thước biểu diễn | 64 |
| C | số lớp | 7 |

```
CSI (R, S, T)
  → chuẩn hóa theo mẫu                               (R, S, T)
  → chia khoảnh khắc W, stride W/2                   (R, S, K, W)
  ├─ nhánh hình dạng: Conv1d mỗi (r, s, k) → token
  │                   → Khối 2: attention R·S token  → h_k  (K, D)
  └─ nhánh tốc độ:    Hann → log|rfft| → trung bình (r, s)
                      → Linear                       → a_k  (K, D)
  → gộp: z_k = Linear(2D → D)([h_k ‖ a_k])  (+ modality dropout trên h)
  → Khối 3: Transformer thời gian, bias có dấu       (D)
  → classifier                                       (C)
```

### 3.2 Chuẩn hóa theo mẫu

Với mỗi mẫu, mô hình:
1. chỉ giữ anten đầu tiên của mỗi RX;
2. trừ trung bình theo thời gian của từng kênh (r, s), để bỏ mức tĩnh của kênh — phần này phụ thuộc môi trường và vị trí, không mang thông tin chuyển động;
3. chia cho **một** độ lệch chuẩn chung cho mỗi (mẫu, RX).

Bước 3 cân bằng giữa các RX nhưng giữ nguyên độ mạnh tương đối giữa các subcarrier, vì kênh nào dao động mạnh hơn thì đó là kênh nhạy hơn. Chuẩn hóa nằm trong mô hình và chỉ dùng thống kê của chính mẫu, nên không phụ thuộc thống kê của tập train.

### 3.3 Khoảnh khắc

Chuỗi được chia thành K = 30 khoảnh khắc dài W = 64 mẫu (0.32 s), chồng nhau 50 %. W là lũy thừa của 2 gần nhất với 0.3·fs. Độ dài này đủ ngắn để mỗi khoảnh khắc gần như chỉ chứa một pha của hành động, và đủ dài để có phổ với bước 3.125 Hz.

### 3.4 Nhánh hình dạng

**Token của từng kênh.** Mỗi đoạn (r, s, k) dài 64 mẫu đi qua:
- 3 lớp Conv1d (1 → 32, k = 8, s = 2; 32 → 64, k = 4, s = 2; 64 → 64, k = 3), mỗi lớp kèm BatchNorm và GELU;
- AdaptiveAvgPool1d(4), giữ 4 ô theo thứ tự thời gian bên trong khoảnh khắc;
- Linear(256 → D), rồi Linear(D → D).

Trọng số dùng chung cho mọi kênh và mọi khoảnh khắc.

**Khối 2: attention RX × subcarrier.** Trong mỗi khoảnh khắc, R·S = 168 token cùng một token CLS đi qua 2 lớp Transformer pre-norm (4 head). Bias attention học được, cộng vào logit trước softmax:
- hai token cùng RX: `b_sub[h, |s − s′|]`, theo khoảng cách subcarrier;
- hai token khác RX: một hằng số `c_cross[h]`;
- CLS có 3 bias riêng.

Mô hình không dùng embedding vị trí tuyệt đối và không có danh tính RX, nên **đầu ra bất biến khi hoán vị RX**. Điều này hợp lý vì vị trí đặt RX thay đổi theo phòng. Đầu ra tại CLS là `h_k`.

### 3.5 Nhánh tốc độ

Với mỗi đoạn (r, s, k):
- trừ trung bình của đoạn;
- nhân với cửa sổ Hann tuần hoàn, đã chia cho trung bình của nó để hệ số khuếch đại bằng cửa sổ chữ nhật;
- tính `log1p|rfft|` và bỏ bin DC, còn 32 bin từ 3.125 đến 100 Hz.

Phổ sau đó được **lấy trung bình qua mọi (r, s)** rồi chiếu Linear(32 → D) thành `a_k`.

**Lý do lấy trung bình:** tần số Doppler do tốc độ cơ thể gây ra là chung cho mọi kênh. Trung bình qua 168 kênh giảm nhiễu mà không mất thông tin tốc độ.

**Lý do dùng Hann:**
- sidelobe −31 dB so với −13 dB của cửa sổ chữ nhật, nên năng lượng ít rò sang bin lân cận;
- với overlap 50 %, tổng các cửa sổ là hằng số, nên không đoạn tín hiệu nào bị giảm trọng số.

Nhánh này chỉ có khoảng 2K tham số.

### 3.6 Gộp theo khoảnh khắc và modality dropout

```
z_k = Linear(2D → D)([h_k ‖ a_k])
```

Gộp tuyến tính trộn **từng chiều riêng**, nên hai nhánh có thể bổ sung cho nhau ở những chiều khác nhau.

Nhánh hình dạng (khoảng 140K tham số) lớn hơn nhánh tốc độ (khoảng 2K) khoảng 70 lần. Nếu train bình thường, Khối 3 có thể chỉ dựa vào nhánh lớn. **Modality dropout:** khi train, toàn bộ `h` của một mẫu, cả 30 khoảnh khắc, bị đặt bằng 0 với xác suất 0.5. Điều này buộc Khối 3 phải đọc được nhánh tốc độ. Lúc test không dùng dropout.

### 3.7 Khối 3: Transformer thời gian với bias có dấu

Chuỗi 30 vector `z_k` cùng một token CLS đi qua 6 lớp Transformer pre-norm (4 head, FFN 4D). Bias attention giữa khoảnh khắc i (query) và j (key) là `b[h, i − j]`, với i − j ∈ {−29, …, 29}, tức 59 giá trị mỗi head. CLS có 3 bias riêng.

Vì `b[h, +d] ≠ b[h, −d]`, mỗi head có thể chuyên nhìn về quá khứ hoặc về tương lai. Bias đối xứng theo `|i − j|` không làm được điều này. Đây là điều kiện cần để phân biệt các hành động có cùng tư thế nhưng ngược thứ tự.

Classifier gồm LayerNorm, Dropout(0.1) và Linear(D → C). Loss là cross-entropy.

### 3.8 Kích thước

| Thành phần | Tham số |
|---|---|
| Nhánh hình dạng (Conv + token + Khối 2) | ≈ 142K |
| Nhánh tốc độ | ≈ 2K |
| Gộp tuyến tính | 8.3K |
| Khối 3 | ≈ 300K |
| **Tổng** | **453,263** |

Mô hình cần **2.378 GMACs/mẫu**, đo bằng `thop`, không tính FFT và phép nhân ma trận bên trong attention. Gần như toàn bộ MACs nằm ở Conv của nhánh hình dạng.

## 4. Thiết lập thí nghiệm

### 4.1 Dữ liệu

SSHAR-ESP, phòng `room_02`:
- 10 người (ID 1, 2, 3, 4, 9, 10, 11, 12, 13, 14);
- 3 bộ thu ESP32, mỗi bộ 56 subcarrier;
- biên độ CSI đã nội suy về 200 Hz, mỗi mẫu 5 s (T = 1000);
- mỗi người có 640 mẫu (80 mẫu cho mỗi hoạt động, thực hiện ở 8 vị trí và nhiều hướng / lần lặp), tổng cộng 6400 mẫu.

**Nhãn: 7 lớp.** Sáu hành động: ngồi xuống, đứng lên, cúi nhặt, ngã, đi, chạy. Lớp thứ bảy là **tĩnh**, gộp "nằm im" và "phòng trống":
- hai trạng thái này không có chuyển động nên gần như không phân biệt được bằng biên độ CSI;
- trong ứng dụng, cả hai đều là "không có hoạt động".

Lớp tĩnh vì thế có 2 × 80 mẫu mỗi người, gấp đôi các lớp khác. Mô hình không dùng class weight.

### 4.2 Giao thức đánh giá: chéo người 5 fold

10 người được ghép thành 5 cặp: (13, 14), (2, 10), (3, 11), (1, 9), (4, 12). Ở fold f, **test là cặp f, val là cặp (f + 1) mod 5**, và 6 người còn lại để train. Kích thước train / val / test là 3840 / 1280 / 1280.

Mỗi người xuất hiện trong tập test đúng một lần và trong tập val đúng một lần. Không người nào vừa ở tập train vừa ở tập test, nên không có rò rỉ thông tin về người.

### 4.3 Thước đo

- **Chính: accuracy trên 6 hành động.**
  - Chỉ tính các mẫu có nhãn thật là hành động; một hành động bị đoán là "tĩnh" vẫn tính là sai.
  - Sáu lớp hành động cân bằng tuyệt đối (160 mẫu mỗi lớp mỗi fold), nên thước đo này bằng balanced accuracy của 6 lớp.
  - Không dùng accuracy 7 lớp làm thước đo chính, vì lớp tĩnh (25 % tập test, gần như luôn đúng) làm số này cao lên và thu hẹp khoảng cách giữa các mô hình.
- **Phụ:** macro-F1 7 lớp, có phạt cả báo động giả từ lớp tĩnh; và accuracy 7 lớp.
- **Kiểm định: McNemar gộp.**
  - Với hai mô hình A, B, đếm trên toàn bộ các fold số mẫu hành động **chỉ A đúng** và **chỉ B đúng**, rồi dùng kiểm định nhị thức chính xác.
  - Bảng báo cáo cả mean ± std qua các fold, nhưng kết luận dựa vào McNemar, vì độ lệch chuẩn giữa các fold (do khác biệt giữa người) thường lớn hơn khoảng cách giữa các mô hình.

### 4.4 Huấn luyện

- Adam, lr đỉnh 1e-3, batch 32, warmup tuyến tính 5 epoch, rồi cosine về 0 ở epoch 100.
- Không dừng sớm.
- Checkpoint là epoch có val accuracy cao nhất (nếu hòa thì giữ epoch sớm nhất). Accuracy test được đo trên checkpoint này.
- Dropout 0.1, seed 42.
- Loader không z-score toàn cục, vì chuẩn hóa nằm trong mô hình.
- Phần cứng: Kaggle T4 hoặc P100. Một run `full` mất khoảng 3 h cho 100 epoch.

### 4.5 Các biến thể ablation

Mỗi biến thể đổi **một** điểm so với `full`:

| Biến thể | Thay đổi | Câu hỏi |
|---|---|---|
| `gate` | Cổng vô hướng mỗi khoảnh khắc: `z = g·LN(h) + (1 − g)·LN(a)` | Gộp tuyến tính hay gộp bằng cổng? |
| `wo_mdrop` | Không modality dropout | Dropout có cần để dùng cả hai nhánh không? |
| `wo_shape` | Chỉ nhánh tốc độ | Nhánh hình dạng có cần không? |
| `wo_speed` | Chỉ nhánh hình dạng | Nhánh tốc độ có cần không? |
| `shape_mean` | Khối 2 = trung bình 168 token | Gộp có chọn lọc qua attention có hơn trung bình không? |
| `wo_order` | Khối 3 = trung bình 30 khoảnh khắc | Thứ tự có cần không? |
| `sym_bias` | Bias Khối 3 đối xứng `b[\|i − j\|]` | Hướng trước/sau có cần không? |
| `no_overlap` | Stride = W (K = 15) | Overlap 50 % có cần không? |
| `no_win` | W = T = 1000 (K = 1) | Chia thành khoảnh khắc có cần không? |

## 5. Kết quả

### 5.1 Kết quả chính

| Fold (cặp test) | 6 hành động | 7 lớp | Epoch tốt nhất | Val acc |
|---|---|---|---|---|
| 0 (13, 14) | 91.67 | 93.75 | 51 | 93.1 |
| 1 (2, 10) | 91.15 | 93.36 | 71 | 97.1 |
| 2 (3, 11) | 92.81 | 94.61 | 28 | 98.4 |
| 3 (1, 9) | 93.12 | 94.84 | 55 | 96.0 |
| 4 (4, 12) | 95.73 | 96.72 | 49 | 92.7 |
| **Trung bình** | **92.90 ± 1.78** | **94.66 ± 1.30** | | |

Macro-F1 7 lớp: **93.81 %**.

**So sánh với baseline** theo cùng giao thức (7 lớp, chéo người 5 fold, thước đo 6 hành động):

| Mô hình | 6 hành động | Macro-F1 7 lớp | Params | MACs |
|---|---|---|---|---|
| WiFormer v5 | 92.90 ± 1.78 | 93.81 | 453K | 2.38 G |
| AMCS-TCN | (bổ sung) | (bổ sung) | | |
| CNN-GRU | (bổ sung) | (bổ sung) | | |
| BiMamba | (bổ sung) | (bổ sung) | | |

Protocol cho các baseline đã được viết trong `amsc-tcn/code/docs/SSHAR_ESP_protocol_v5.md`.

### 5.2 Ablation

Do quota GPU, các ablation hoàn thành theo đợt. Mỗi bảng chỉ so trên **các fold mà mọi biến thể trong bảng đều có**. Quy ước chung cho các bảng:
- Δ: chênh lệch accuracy 6 hành động so với `full` trên cùng các fold.
- Lỗi tại chỗ / lỗi di chuyển: số mẫu hành động bị đoán nhầm sang lớp khác **trong cùng nhóm** (ngồi / đứng / cúi / ngã, và đi / chạy).
- "full đúng / B đúng": số mẫu chỉ `full` đúng và số mẫu chỉ biến thể B đúng.
- p: McNemar gộp so với `full`.

**Bảng 1. Đủ 5 fold (4800 mẫu hành động).**

| Biến thể | F0 | F1 | F2 | F3 | F4 | 6 hành động | Macro-F1 | Δ | Lỗi tại chỗ | Lỗi di chuyển | full đúng / B đúng | p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **full** | 91.67 | 91.15 | 92.81 | 93.12 | 95.73 | **92.90 ± 1.78** | **93.81** | | **134** | 180 | | |
| sym_bias | 86.15 | 90.83 | 92.29 | 92.60 | 91.77 | 90.73 ± 2.65 | 91.93 | −2.17 | 239 | 172 | 206 / 102 | < 1e-4 |
| gate | 87.19 | 92.19 | 92.08 | 90.31 | 89.48 | 90.25 ± 2.07 | 91.55 | −2.65 | 265 | 200 | 250 / 123 | < 1e-4 |
| wo_shape | 84.38 | 90.73 | 89.90 | 86.67 | 93.23 | 88.98 ± 3.48 | 90.43 | −3.92 | 267 | 253 | 292 / 104 | < 1e-4 |
| no_win | 86.46 | 83.23 | 86.88 | 90.94 | 85.94 | 86.69 ± 2.77 | 88.54 | −6.21 | 445 | 187 | 458 / 160 | < 1e-4 |
| wo_order | 85.42 | 81.88 | 85.62 | 87.40 | 84.79 | 85.02 ± 2.01 | 86.93 | −7.88 | 499 | 210 | 486 / 108 | < 1e-4 |

**Bảng 2. Fold 0–2 (2880 mẫu hành động).**

| Biến thể | F0 | F1 | F2 | 6 hành động | Macro-F1 | Δ | Lỗi tại chỗ | Lỗi di chuyển | p |
|---|---|---|---|---|---|---|---|---|---|
| **full** | 91.67 | 91.15 | 92.81 | **91.88** | **92.92** | | **76** | 158 | |
| gate | 87.19 | 92.19 | 92.08 | 90.49 | 91.73 | −1.39 | 123 | 151 | 0.004 |
| sym_bias | 86.15 | 90.83 | 92.29 | 89.76 | 91.08 | −2.12 | 141 | 149 | < 1e-4 |
| no_overlap | 86.77 | 89.90 | 91.98 | 89.55 | 90.87 | −2.33 | 135 | 166 | < 1e-4 |
| wo_speed | 85.62 | 89.48 | 92.92 | 89.34 | 90.79 | −2.53 | 136 | 166 | < 1e-4 |
| shape_mean | 84.58 | 91.25 | 91.56 | 89.13 | 90.61 | −2.74 | 146 | 160 | < 1e-4 |
| wo_shape | 84.38 | 90.73 | 89.90 | 88.33 | 89.75 | −3.54 | 144 | 189 | < 1e-4 |
| no_win | 86.46 | 83.23 | 86.88 | 85.52 | 87.51 | −6.35 | 240 | 176 | < 1e-4 |
| wo_order | 85.42 | 81.88 | 85.62 | 84.31 | 86.25 | −7.57 | 275 | 175 | < 1e-4 |

**Bảng 3. Fold 0–1, đủ 10 biến thể (1920 mẫu hành động).**

| Biến thể | 6 hành động | Macro-F1 | Δ | Lỗi tại chỗ | Lỗi di chuyển | full đúng / B đúng | p |
|---|---|---|---|---|---|---|---|
| **full** | **91.41** | **92.53** | | **62** | 103 | | |
| gate | 89.69 | 91.04 | −1.72 | 92 | 106 | 76 / 43 | 0.003 |
| sym_bias | 88.49 | 89.94 | −2.92 | 117 | 100 | 98 / 42 | < 1e-4 |
| no_overlap | 88.33 | 89.84 | −3.07 | 110 | 114 | 89 / 30 | < 1e-4 |
| wo_mdrop | 88.12 | 89.79 | −3.28 | 117 | 109 | 100 / 37 | < 1e-4 |
| shape_mean | 87.92 | 89.61 | −3.49 | 112 | 115 | 94 / 27 | < 1e-4 |
| wo_shape | 87.55 | 89.15 | −3.85 | 120 | 116 | 112 / 38 | < 1e-4 |
| wo_speed | 87.55 | 89.23 | −3.85 | 118 | 117 | 104 / 30 | < 1e-4 |
| no_win | 84.84 | 86.88 | −6.56 | 163 | 127 | 165 / 39 | < 1e-4 |
| wo_order | 83.65 | 85.68 | −7.76 | 197 | 116 | 186 / 37 | < 1e-4 |

**Nhận xét:**

1. **Mọi thành phần đều có đóng góp có ý nghĩa thống kê.** Mọi biến thể đều kém `full` với p ≤ 0.004 trên mọi tập fold có sẵn.
2. **Có hai mức ảnh hưởng tách bạch.**
   - Cấu trúc thời gian quan trọng nhất: `wo_order` mất 7.9 điểm, `no_win` mất 6.2 điểm. Lỗi tại chỗ tăng 3.7 và 3.3 lần.
   - Các thành phần còn lại (cách gộp, hướng thời gian, hai nhánh, Khối 2, overlap, dropout) mỗi thành phần đóng góp 1.4–3.9 điểm.
3. **Lợi ích nằm ở các hành động tại chỗ.** Lỗi tại chỗ của `full` thấp nhất ở mọi bảng, và mọi ablation làm nó tăng 1.5–3.7 lần. Ngược lại, lỗi di chuyển (đi ↔ chạy) gần như không đổi, khoảng 150–210 trên 5 fold. Ngoại lệ duy nhất là `wo_shape` với 253 lỗi. Mục 5.7 giải thích hiện tượng này.
4. **Mức chênh co lại khi thêm fold** đối với `wo_speed` và `shape_mean`: khoảng −3.5…−3.9 ở fold 0–1, còn −2.5…−2.7 ở fold 0–2. Độ lớn cuối cùng của hai mức chênh này cần đủ 5 fold. Mức chênh của `sym_bias` ổn định quanh −2.2 điểm.

### 5.3 Phân tích theo nhóm hành động

Chia các lớp thành ba nhóm: **tại chỗ** (ngồi, đứng, cúi, ngã), **di chuyển** (đi, chạy) và **tĩnh**.

- **Nhầm giữa các nhóm rất hiếm.** Với `full` trên 5 fold, chỉ 28/6400 mẫu (0.4 %) bị đoán sang nhóm khác, nên phân loại theo nhóm đúng khoảng 99.6 %.
- Trong 28 lỗi đó, 21 là các mẫu chạy của người 1 bị đoán thành cúi, ngồi hoặc ngã. Đây là bất thường riêng của người này.
- **Lớp tĩnh tách biệt hoàn toàn:** 1599/1600 mẫu đúng. Chỉ 1 mẫu hành động bị đoán thành tĩnh và 1 mẫu tĩnh bị đoán thành hành động.

Vì vậy bài toán thực chất là phân biệt **bên trong** nhóm tại chỗ và **bên trong** nhóm di chuyển. Các ablation ở mục 5.2 cho thấy kiến trúc chủ yếu cải thiện phần thứ nhất.

### 5.4 Confusion matrix

**Bảng 4.** `full`, cộng 5 fold. Hàng là nhãn thật, cột là dự đoán. Mỗi lớp hành động có 800 mẫu, lớp tĩnh 1600 mẫu.

| | ngồi | đứng | cúi | ngã | đi | chạy | tĩnh | Recall |
|---|---|---|---|---|---|---|---|---|
| **ngồi** | 733 | 36 | 10 | 21 | 0 | 0 | 0 | 91.6 |
| **đứng** | 19 | 761 | 20 | 0 | 0 | 0 | 0 | 95.1 |
| **cúi** | 2 | 8 | 789 | 1 | 0 | 0 | 0 | 98.6 |
| **ngã** | 16 | 0 | 1 | 781 | 0 | 2 | 0 | 97.6 |
| **đi** | 0 | 0 | 0 | 0 | 786 | 14 | 0 | 98.3 |
| **chạy** | 7 | 0 | 14 | 3 | **166** | 609 | 1 | **76.1** |
| **tĩnh** | 0 | 1 | 0 | 0 | 0 | 0 | 1599 | 99.9 |
| **Precision** | 94.3 | 94.4 | 94.6 | 96.9 | 82.6 | 97.4 | 99.9 | |

- **Chạy → đi** là lỗi lớn nhất (166 mẫu, 49 % tổng lỗi), và gần như chỉ theo một chiều: đi → chạy chỉ có 14 mẫu.
- **Ngồi xuống là lớp tại chỗ khó nhất (91.6 %).**
  - Ngồi → đứng (36) nhiều gấp đôi đứng → ngồi (19): một phần động tác bị đọc ngược chiều.
  - Ngồi ↔ ngã có 37 lỗi, rơi vào những người khác nhau tùy theo chiều.
- **Cúi nhặt gần như hoàn hảo** (98.6 %), dù cũng là động tác lên xuống.

**Mỗi ablation để lại "dấu vết" riêng**, so trên cùng fold với `full`:

| Biến thể | Lỗi tăng rõ nhất | Diễn giải |
|---|---|---|
| `wo_order` | ngồi ↔ đứng: 235 so với 51 (f0–3) | Mất thứ tự thì không phân biệt được hai động tác ngược chiều nhau |
| `no_win` | ngồi ↔ đứng: 202 so với 51 (f0–3); cúi → đứng: 40 so với 5 | Một vector cho cả 5 s làm mất diễn biến theo thời gian |
| `sym_bias` | ngồi → ngã: 44 so với 13 (f0–3) | Không phân biệt trước/sau thì nhầm "hạ người từ từ" với "hạ người đột ngột" |
| `wo_shape` | ngồi → đứng: 73 so với 28 (f0–1) | Hình dạng giúp đọc chiều chuyển động |
| `wo_speed`, `shape_mean`, `wo_mdrop` | ngã → ngồi: 33–36 so với 16 (f0–1) | Ngã khác ngồi chủ yếu ở tốc độ |

### 5.5 Mô hình đã học gì

**Hai nhánh đều được dùng.** Bảng dưới là knock-out: đặt một nhánh bằng 0 lúc test, trên mô hình đã train, tính accuracy 7 lớp.

| Fold | Đủ hai nhánh | Chỉ tốc độ | Chỉ hình dạng |
|---|---|---|---|
| 0 | 93.8 | 87.8 | 44.9 |
| 1 | 93.4 | 93.5 | 80.2 |
| 2 | 94.6 | 94.6 | 66.0 |
| 3 | 94.8 | 89.0 | 64.9 |
| 4 | 96.7 | 93.4 | 32.1 |

- Nhờ modality dropout, mô hình vẫn hoạt động tốt khi chỉ còn nhánh tốc độ.
- Có đủ hai nhánh thì tốt hơn rõ ở fold 0, 3 và 4 (+3.3 đến +6.0 điểm).
- Nếu chỉ còn nhánh hình dạng thì mô hình kém hẳn. Lý do là nó chưa bao giờ được train trong tình huống thiếu nhánh tốc độ.
- Chuẩn của trọng số gộp: khoảng 5.8 cho nửa hình dạng và khoảng 9.0 cho nửa tốc độ, ổn định trên 5 fold.

**Nhánh tốc độ quyết định việc tách đi với chạy.** Nếu chỉ giữ nhánh hình dạng, recall chạy chỉ còn 0–3 % ở fold 2–4. Ngược lại, nhánh hình dạng giúp ổn định lớp đi: ở fold 4, chỉ dùng nhánh tốc độ thì recall đi tụt xuống 65 %.

**Khối 3 học được hướng thời gian, nhất quán trên mọi fold.** Bảng dưới là bias học được `b[h, i − j]`. Offset i − j < 0 nghĩa là key nằm **sau** query.

| Fold | Bias TB, quá khứ (offset +1…+5), 4 head | Bias TB, tương lai (offset −5…−1), 4 head | Offset có bias lớn nhất của từng head |
|---|---|---|---|
| 0 | −0.84, −0.97, −0.76, −0.73 | +0.85, +0.43, +0.59, −0.13 | −4, −8, −5, −9 |
| 1 | −1.15, −1.10, −0.47, −0.67 | +0.94, +0.42, −0.10, +0.02 | −5, −7, −9, −9 |
| 2 | −1.01, −0.68, −0.77, −0.48 | +0.77, +0.39, −0.15, −0.26 | −4, −7, −9, −10 |
| 3 | −1.31, −1.09, −0.82, −0.41 | +0.86, +0.50, −0.08, +0.27 | −5, −7, −8, −4 |
| 4 | −1.19, −0.85, −0.92, −0.54 | +0.64, +0.13, −0.23, +0.20 | −5, −8, −9, −5 |

- Ở cả 5 fold, quá khứ gần bị **hạ trọng số** (bias khoảng −0.5 đến −1.3), và mọi head đạt bias lớn nhất ở một offset **tương lai** từ 4 đến 10 khoảnh khắc. Với bước 0.16 s, khoảng này tương ứng 0.64–1.6 s sau.
- Nói cách khác, mỗi khoảnh khắc "nhìn trước" pha tiếp theo của hành động. Bias đối xứng (`sym_bias`) không thể biểu diễn kiểu nhìn này, và việc nó kém 2.2 điểm xác nhận điều đó.

**Khối 2 dựa vào nội dung, không dựa vào vị trí.** Bias học được của Khối 2 rất nhỏ: |b_sub| < 0.12 và |c_cross| < 0.05 ở mọi head. Attention RX × subcarrier vì thế gần như hoàn toàn do nội dung token quyết định (Q·K), không phải do khoảng cách subcarrier. Dù vậy, attention vẫn tốt hơn lấy trung bình 168 token (`shape_mean`, −2.7 đến −3.5 điểm).

**Bất biến khi hoán vị RX được xác nhận.** Hoán vị vòng 3 RX lúc test cho dự đoán trùng 100 % ở cả 5 fold, và logit lệch nhau tối đa 2·10⁻⁵, chỉ là sai số số học.

### 5.6 Vì sao gộp bằng cổng thất bại

Với `gate`, mỗi khoảnh khắc có một cổng vô hướng: `z_k = g_k·LN(h_k) + (1 − g_k)·LN(a_k)`. Thiết kế này hấp dẫn vì `g_k` đọc trực tiếp được như tỉ trọng của nhánh hình dạng, nhưng kết quả kém hơn gộp tuyến tính 2.65 điểm (p < 1e-4).

Giá trị g trung bình trên tập test, 5 fold:

| Tất cả | ngồi | đứng | cúi | ngã | đi | chạy | tĩnh |
|---|---|---|---|---|---|---|---|
| **0.068** | 0.071 | 0.072 | 0.088 | 0.064 | 0.087 | 0.073 | 0.045 |

- **Cổng sập về nhánh tốc độ:** khoảng 93 % đầu ra lấy từ nhánh tốc độ, ở mọi lớp.
- Knock-out xác nhận điều này: bỏ nhánh hình dạng gần như không đổi kết quả (ví dụ fold 0: 90.4 → 90.3), còn nếu chỉ giữ nhánh hình dạng thì accuracy chỉ 15–50 %.
- Hệ quả là lỗi tại chỗ tăng gấp đôi (265 so với 134).

**Giải thích:**
1. Modality dropout làm `h' = 0` ở một nửa số mẫu train. Với `z = g·h' + (1 − g)·a'`, khi h' bằng 0 thì g càng nhỏ, a' càng ít bị thu nhỏ, nên cổng có lợi khi luôn giữ g thấp.
2. Khi g đã nhỏ, gradient về nhánh hình dạng cũng bị nhân với g. Nhánh hình dạng học chậm, và cổng càng ít có lý do chọn nó: một vòng tự củng cố, đúng là hiện tượng học "tham lam" theo nguồn dễ (Wang et al., CVPR 2020).

Gộp tuyến tính không gặp vấn đề này vì mỗi chiều đầu ra có trọng số riêng cho cả hai nhánh. Một cách gộp giải thích được bằng một con số duy nhất vì thế không tương thích với modality dropout trong bài toán này.

### 5.7 Đi ↔ chạy: lỗi do người, không do kiến trúc

**Lỗi tập trung vào ba người.** Bảng dưới là kết quả của `full`; mỗi người có 80 mẫu đi và 80 mẫu chạy.

| Người | Recall chạy | Chạy → đi |
|---|---|---|
| 10 | **8.8 %** | 73 |
| 11 | **43.8 %** | 45 |
| 14 | **65.0 %** | 28 |
| 3 | 87.5 % | 10 |
| Sáu người còn lại | 92.5–100 % (trừ người 1, xem dưới) | 0–6 |

- Ba người 10, 11 và 14 chiếm 146/166 (88 %) lỗi chạy → đi.
- Lỗi trải đều theo vị trí (14–28 lỗi mỗi vị trí trong 8 vị trí), theo hướng (88 / 78) và theo lần lặp (30–39).
- Người 1 có recall chạy 66.2 %, nhưng phần lớn lỗi là chạy bị đoán sang nhóm tại chỗ (21 mẫu), không phải sang đi.

Như vậy lỗi không đến từ hình học phòng mà đến từ **cách chạy của từng người**.

**Phổ tốc độ.** Trọng tâm phổ (spectral centroid) được tính đúng như nhánh tốc độ của mô hình: chuẩn hóa theo mẫu, khoảnh khắc 64 mẫu, Hann, `log1p|rfft|` bỏ DC, trung bình qua (r, s, k). Mỗi giá trị là trung bình trên 80 mẫu của một người.

| Người | Đi (Hz) | Chạy (Hz) | Khoảng cách chạy − đi | Recall chạy |
|---|---|---|---|---|
| 9 | 26.6 | 34.4 | 7.9 | 92.5 % |
| 1 | 30.8 | 36.6 | 5.8 | 66.2 %* |
| 13 | 26.1 | 31.7 | 5.6 | 100 % |
| 4 | 30.2 | 34.7 | 4.5 | 98.8 % |
| 2 | 31.1 | 35.4 | 4.3 | 100 % |
| 12 | 31.6 | 34.7 | 3.0 | 98.8 % |
| 3 | 31.1 | 33.2 | 2.1 | 87.5 % |
| 11 | 30.3 | 32.2 | **1.9** | 43.8 % |
| 14 | 29.7 | 31.2 | **1.5** | 65.0 % |
| 10 | 27.3 | 28.6 | **1.3** | 8.8 % |

\* Người 1: lỗi chủ yếu là chạy → nhóm tại chỗ.

- **Bốn người có khoảng cách chạy − đi nhỏ nhất (≤ 2.1 Hz) cũng là bốn người có nhiều lỗi chạy → đi nhất.**
- Người 10 "chạy" với trọng tâm phổ 28.6 Hz, thấp hơn trung bình lúc **đi** của cả nhóm (29.5 Hz).
- Với những người này, khoảng cách đi/chạy nhỏ hơn độ dao động giữa các mẫu của chính người đó (độ lệch chuẩn 1.3–2.2 Hz). Kể cả khi biết ngưỡng riêng của người 10, một ngưỡng tốc độ duy nhất vẫn sai khoảng 1/3 số mẫu.
- Mô hình chỉ học được ngưỡng chung (khoảng 31 Hz) từ những người khác, nên gần như mọi mẫu chạy của người 10 đều rơi vào phía "đi".
- Mô hình **sai mà rất tự tin**: với mẫu chạy bị đoán thành đi, xác suất "đi" trung bình là 0.92, và 119/166 mẫu có xác suất > 0.9. Đây là dấu hiệu của khác biệt giữa người (domain shift), không phải của mẫu nằm sát ranh giới.

Trọng tâm phổ chỉ là một chỉ số thô dùng để so tương đối giữa các người, không phải tần số Doppler thật.

**Vì sao ablation không ảnh hưởng tới lỗi này:** tách đi với chạy dựa gần như hoàn toàn vào nhánh tốc độ (mục 5.5). Khi tốc độ của một người không khác biệt giữa hai lớp, không thành phần nào của kiến trúc có thêm manh mối. Một manh mối có thể có là **nhịp bước** (khoảng 1.8–2 bước/s khi đi, 2.6–3.2 bước/s khi chạy). Tuy nhiên:
- nhịp bước nằm dưới bin phổ đầu tiên (3.125 Hz) của mỗi khoảnh khắc;
- các khoảnh khắc cách nhau 0.16 s, nên Khối 3 chỉ thấy được biến đổi tới khoảng 3.1 Hz, sát với nhịp chạy.

Đây mới là giả thuyết, chưa được kiểm chứng.

**Không tính lỗi đi/chạy của ba người này** (chỉ dùng để chẩn đoán, không phải kết quả báo cáo): `full` đạt **95.53 %** accuracy 6 hành động, còn 193 lỗi trên 4320 mẫu. Lỗi còn lại phân bố như sau:
- 68 % là lỗi tại chỗ: ngồi ↔ đứng 55, ngã ↔ ngồi 37, cúi ↔ đứng 28, cúi ↔ ngồi 12;
- đi ↔ chạy chỉ 32, khoảng 3 % số mẫu đi/chạy của 7 người còn lại.

### 5.8 Độ khó theo từng người

Bảng dưới là kết quả `full`; mỗi người có 480 mẫu hành động.

| Người | Fold | Acc 6 hành động | Lỗi tại chỗ | Lỗi đi ↔ chạy | Chạy → tại chỗ |
|---|---|---|---|---|---|
| 2 | 1 | 98.5 | 7 | 0 | 0 |
| 12 | 4 | 97.7 | 10 | 1 | 0 |
| 3 | 2 | 97.5 | 2 | 10 | 0 |
| 13 | 0 | 94.6 | 26 | 0 | 0 |
| 1 | 3 | 94.2 | 1 | 6 | 21 |
| 4 | 4 | 93.8 | 22 | 7 | 1 |
| 9 | 3 | 92.1 | 27 | 8 | 3 |
| 14 | 0 | 88.8 | 24 | 30 | 0 |
| 11 | 2 | 88.1 | 12 | 45 | 0 |
| 10 | 1 | 83.8 | 5 | 73 | 0 |

Accuracy theo người dao động từ 83.8 đến 98.5 %. Độ dao động giữa các fold (độ lệch chuẩn 1.8 điểm) chủ yếu đến từ ba người chạy chậm và từ lỗi tại chỗ của các người 9, 13, 14, 4.

## 6. Thảo luận và hạn chế

- **Một seed.** Mỗi cấu hình mới train một lần. McNemar chỉ tính đến sự ngẫu nhiên của tập test, không tính đến sự ngẫu nhiên khi train. Các mức chênh lớn (`wo_order`, `no_win`, 6–8 điểm) là chắc chắn. Các mức chênh 1.4–2.7 điểm (`gate`, `sym_bias`, `wo_speed`, `shape_mean`) cần thêm seed của `full` để ước lượng độ dao động khi train. (bổ sung)
- **Ablation chưa đủ fold.** `wo_mdrop` có 2/5 fold; `wo_speed`, `shape_mean`, `no_overlap` có 3/5 fold. (bổ sung)
- **Chọn checkpoint theo val khá nhiễu.** Train accuracy đạt khoảng 100 % ở mọi run, và epoch tốt nhất dao động từ 19 đến 71. Tập val chỉ gồm 2 người, nên checkpoint được chọn phụ thuộc nhiều vào hai người đó.
- **Một phòng.** Mọi kết quả là trên `room_02`. Có thể đánh giá chuyển phòng mà không cần train lại: chạy mô hình trên `room_01`, chỉ dùng 4 người mới (5–8) để vừa khác phòng vừa khác người. (bổ sung)
- **Đi ↔ chạy với người chạy chậm** là giới hạn chính. Muốn cải thiện có lẽ cần manh mối khác ngoài tốc độ (nhịp bước) hoặc thích nghi theo người, không phải phổ mịn hơn: `no_win`, với phổ mịn gấp 15 lần, không cải thiện được lớp chạy.
- **Mô hình quá tự tin khi sai.** 52 % lỗi có xác suất > 0.9, nên ngưỡng tin cậy không lọc được lỗi do khác biệt giữa người.
- **Cửa sổ Hann** được chọn theo lập luận xử lý tín hiệu. Thử nghiệm sơ bộ (1 fold, cấu hình chỉ có nhánh tốc độ, train 8 lớp) cho thấy Hann tốt hơn cửa sổ chữ nhật khoảng 3 điểm (p = 0.014). Chưa có ablation theo đúng giao thức 7 lớp, 5 fold.
- **Độ dài khoảnh khắc** mới chỉ được kiểm tra ở hai điểm cực (`no_overlap`, `no_win`). Chưa có khảo sát độ nhạy với W = 32 và W = 128.

## 7. Kết luận

WiFormer v5 mô hình hóa một hoạt động như một chuỗi khoảnh khắc ngắn. Trong mỗi khoảnh khắc, mô hình tách thông tin riêng của từng kênh (hình dạng, gộp có chọn lọc bằng attention RX × subcarrier, bất biến khi hoán vị RX) khỏi thông tin chung của mọi kênh (tốc độ, từ phổ Hann trung bình). Hai nguồn được gộp tuyến tính với modality dropout, rồi đọc theo thứ tự bằng Transformer thời gian với bias có dấu.

Trên SSHAR-ESP với đánh giá chéo người 5 fold, mô hình đạt **92.90 %** accuracy 6 hành động. Mọi thành phần đều có đóng góp có ý nghĩa thống kê, và mỗi thành phần sửa đúng nhóm lỗi nó nhắm tới:
- cấu trúc thời gian cho ngồi ↔ đứng;
- tốc độ cho ngã ↔ ngồi và đi ↔ chạy;
- hình dạng cho chiều của chuyển động.

Phân tích cũng chỉ ra hai điều:
- gộp bằng cổng vô hướng sập về nhánh nhỏ khi dùng cùng modality dropout;
- lỗi chính còn lại (chạy → đi) đến từ những người chạy với tốc độ gần như đi bộ, tức là giới hạn của việc tổng quát hóa sang người mới chứ không phải của kiến trúc.

---

## Phụ lục A. Số tham số và thời gian train

Thời gian là `training_time` của 100 epoch trên Kaggle, trung bình fold 0–1.

| Biến thể | Params | Thời gian |
|---|---|---|
| full | 453,263 | ~3.0 h |
| gate | 445,136 | ~3.0 h |
| wo_mdrop | 453,263 | ~3.1 h |
| wo_shape | 307,071 | ~0.1 h |
| wo_speed | 447,055 | ~3.1 h |
| shape_mean | 352,991 | ~2.4 h |
| wo_order | 153,047 | ~3.3 h |
| sym_bias | 453,147 | ~3.3 h |
| no_overlap | 453,143 | ~1.65 h |
| no_win | 482,983 | ~1.14 h |

## Phụ lục B. Tái lập

- Code: [models/wiformer_v5.py](models/wiformer_v5.py), nhánh `wiformer_v5`.
- Mô tả kỹ thuật: [WiFormer_v5_Architecture.md](WiFormer_v5_Architecture.md).
- Lệnh chạy:

```
python ablation.py --model=wiformer_v5 --data_root=<thư mục chỉ chứa room_02> --dataset=sshar_esp \
    --split_mode=user --case_split=<fold 0..4> --variants full gate wo_shape ...
python summarize_folds.py --root results/v5 --variants full gate wo_shape wo_mdrop wo_speed \
    shape_mean wo_order sym_bias no_overlap no_win
```

- Phân tích đi/chạy: `results/v5/summary/walk_run_speed.md` (thư mục `results/` không nằm trong git).
- Hướng dẫn chạy trên Kaggle: [kaggle_cells.txt](kaggle_cells.txt).
