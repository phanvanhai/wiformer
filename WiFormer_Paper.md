# WiFormer: Transformer phân cấp với token hai góc nhìn thời gian–phổ cho nhận dạng hoạt động bằng CSI WiFi

*Bản thảo nội bộ. Kết quả: SSHAR-ESP (split theo lần lặp) và UT-HAR, mỗi cấu hình 1 seed. Các ô "(bổ sung)" sẽ được điền khi có kết quả.*

---

## 1. Tóm tắt ý tưởng

CSI biên độ của một mẫu hoạt động là một khối ba chiều **RX × subcarrier × thời gian**. Ba chiều này mang ba loại thông tin khác nhau:
- trong một đoạn ngắn của một kênh: hình dạng và tốc độ dao động;
- giữa các kênh trong cùng một đoạn: kênh nào nhạy với cử động;
- giữa các đoạn: diễn biến của hành động.

WiFormer xử lý lần lượt ba thang đó bằng ba khối, với ba ý tưởng chính:

1. **Token hai góc nhìn, gộp sớm.** Mỗi đoạn ngắn của mỗi kênh (rx, subcarrier) được mô tả đồng thời bằng dạng sóng (Conv1d) và phổ biên độ (FFT). Hai mô tả được gộp ngay thành một token. Dạng sóng và phổ là hai góc nhìn **thẳng hàng** của cùng một đoạn tín hiệu, nên gộp tại token cho phép chúng tương tác ở mọi tầng sau.
2. **Attention không gian với bias vị trí tương đối, không phụ thuộc danh tính RX.** Trong mỗi đoạn, attention giữa các token (rx, subcarrier) dùng bias theo khoảng cách subcarrier và một bias cho cặp khác RX. Không có embedding vị trí tuyệt đối.
3. **Attention thời gian với bias tương đối có dấu.** Giữa các đoạn, bias attention phụ thuộc vào hiệu `i − j` có dấu, nên phân biệt được "trước" với "sau". Điều này cần cho các hành động đối xứng gương theo thời gian, như ngồi xuống và đứng lên.

Ngoài ra, mô hình chuẩn hóa từng mẫu để bỏ mức tĩnh của từng kênh (dấu vân tay của môi trường và người), nhưng vẫn giữ độ mạnh tương đối giữa các subcarrier.

Trên SSHAR-ESP (8 lớp, 3 RX ESP32), WiFormer đạt **96.80%** độ chính xác (6 hành động: **97.29%**), cao hơn CNN-GRU và BiMamba 1.5–1.9 điểm (6 hành động: 3.4–4.1 điểm). Ablation cho thấy mỗi thành phần đều có đóng góp, và mỗi thành phần ứng với một nhóm lỗi riêng, khớp với lý do thiết kế.

## 2. Các nghiên cứu liên quan

**Nhận dạng hoạt động bằng CSI WiFi.**
- Các hệ thống ban đầu dựa trên mô hình vật lý và đặc trưng thủ công. CARM (Wang et al., MobiCom 2015) liên hệ tốc độ thay đổi của CSI với tốc độ chuyển động thông qua phân tích tần số theo thời gian.
- Widar3.0 (Zheng et al., MobiSys 2019) dùng phổ Doppler và đặc trưng độc lập với vị trí để tổng quát sang miền mới.
- Các mô hình học sâu (CNN, LSTM, CNN-GRU, ViT) được tổng hợp và so sánh trong SenseFi (Yang et al., Patterns 2023), cùng bộ dữ liệu UT-HAR (Yousefi et al., IEEE Commun. Mag. 2017).
- Nhìn chung, các mô hình này coi CSI là ảnh hoặc chuỗi đa kênh, và chỉ dùng một trong hai miền: thời gian, hoặc phổ (spectrogram).

**Transformer cho CSI.**
- THAT (Li et al., AAAI 2021) dùng hai luồng Transformer, một theo thời gian và một theo kênh, rồi gộp ở cuối.
- Các biến thể ViT chia CSI thành patch 2D.
- Đa số coi subcarrier như một trục ảnh, dùng embedding vị trí tuyệt đối, và không tách riêng chiều RX.

**Mã hóa vị trí tương đối.**
- Shaw et al. (NAACL 2018) đưa vị trí tương đối vào attention. T5 (Raffel et al., JMLR 2020) và Swin (Liu et al., ICCV 2021) dùng bias vô hướng học được theo khoảng cách.
- ALiBi (Press et al., ICLR 2022) dùng bias tuyến tính cố định theo `|i − j|`, đối xứng hai chiều.
- WiFormer dùng bias học được, **có dấu** theo trục thời gian, và **theo khoảng cách** theo trục subcarrier.

**Kết hợp miền thời gian và miền tần số.** Các mô hình CSI thường chọn một trong hai miền: chuỗi biên độ theo thời gian, hoặc spectrogram. Gộp đặc trưng ngay ở mức đầu vào thích hợp khi các nguồn thẳng hàng về không gian và thời gian. Thời gian và phổ của cùng một đoạn CSI là trường hợp như vậy.

**Baseline.** CNN-GRU (CNN trích đặc trưng không gian theo từng bước thời gian, GRU mô hình hóa chuỗi) và BiMamba (mô hình không gian trạng thái hai chiều, dựa trên Mamba của Gu & Dao, 2023).

## 3. Kiến trúc

Ký hiệu: R RX, S subcarrier, T mẫu thời gian, tần số lấy mẫu fs, cửa sổ W, bước W/2, K đoạn, kích thước token D, C lớp.

```
CSI (R, S, T)
  → Chuẩn hóa mẫu                         (R, S, T)
  → Cắt đoạn W, bước W/2                  (R, S, K, W)
  → Khối 1: token hai góc nhìn            (R, S, K, D)
  → Khối 2: attention RX × subcarrier     (K, D)       [mỗi đoạn, lấy CLS]
  → Khối 3: attention thời gian           (D)          [lấy CLS]
  → Classifier                            (C)
```

### 3.1 Chuẩn hóa mẫu

Với mỗi mẫu: trừ trung bình theo thời gian của từng kênh (rx, s), rồi chia **một** độ lệch chuẩn chung cho mỗi RX:

```
x̃[r, s, t] = (x[r, s, t] − mean_t x[r, s, ·]) / std_{s,t}(x[r, ·, ·] − mean)
```

- Trừ trung bình từng kênh: bỏ mức tĩnh. Mức tĩnh phụ thuộc vào môi trường, vị trí và người, không phụ thuộc vào hành động.
- Chia một độ lệch chuẩn chung cho cả RX, thay vì cho từng kênh: giữ **độ mạnh tương đối** giữa các subcarrier. Subcarrier nhạy với cử động có dao động lớn hơn, và thông tin này không bị san phẳng.
- Cái giá: hai trạng thái tĩnh (người nằm im và phòng trống) chỉ còn phân biệt được qua các dao động rất nhỏ.

### 3.2 Khối 1: token hai góc nhìn

Mỗi đoạn `u ∈ ℝ^W` của một kênh (rx, s) được mô tả bằng hai vector. Trọng số dùng chung cho mọi kênh và mọi đoạn.

- **Góc nhìn thời gian:** 3 lớp Conv1d (kênh 1→32→64→64, kernel 8/4/3, stride 2/2/1, BatchNorm, GELU), rồi `AdaptiveAvgPool1d(4)` giữ 4 bước thời gian trong đoạn, rồi `Linear(4·64 → D)`, cho `v_time`.
- **Góc nhìn phổ:** `rfft(u)`, lấy `log(1 + |·|)`, bỏ thành phần DC, giữ W/2 bin, rồi `Linear(W/2 → D)`, cho `v_freq`. Đây là một cột của spectrogram STFT (cửa sổ chữ nhật) của riêng kênh đó.
- **Gộp sớm:** `token = Linear(2D → D)([v_time ‖ v_freq])`.

W được chọn là lũy thừa của 2 gần nhất với 0.3·fs, để các lớp Conv stride 2 và lớp pool chia đoạn thành các phần bằng nhau.

### 3.3 Khối 2: attention RX × subcarrier trong từng đoạn

- Với mỗi đoạn k, lấy R·S token và thêm một token CLS, rồi cho qua L₂ lớp Transformer (pre-norm).
- Attention của mỗi head h có thêm bias:
  - cùng RX: `b_h[|s − s′|]`, theo khoảng cách subcarrier;
  - khác RX: một số vô hướng `c_h`;
  - cặp có CLS: ba bias riêng (CLS→token, token→CLS, CLS→CLS). CLS tách khỏi mọi vị trí.
- Đầu ra là CLS của đoạn k, gọi là `h_k ∈ ℝ^D`.

Thiết kế này có hai tính chất:
1. Mô hình học được kênh nào đáng chú ý, thay vì lấy trung bình đều.
2. Mô hình **bất biến khi hoán vị RX**, vì không có danh tính RX.

### 3.4 Khối 3: attention thời gian

- Chuỗi `h_1..h_K` cùng một CLS đi qua L₃ lớp Transformer.
- Bias của head h là `β_h[i − j]`, học được cho mọi hiệu có dấu từ −(K−1) đến K−1. CLS cũng có ba bias riêng.
- CLS cuối đi qua `LayerNorm → Dropout → Linear(D → C)`. Hàm mất mát là cross-entropy.

### 3.5 Các giả thuyết thiết kế

| | Giả thuyết | Kiểm tra bằng |
|---|---|---|
| H1 | Trích đặc trưng trong đoạn là cần thiết; phép chiếu tuyến tính đoạn thô không đủ | `wo_block1` |
| H2 | Hai góc nhìn bổ sung nhau. Phổ mang tốc độ (đi ↔ chạy). Dạng sóng mang hình dạng (chuyển tư thế, phân biệt trạng thái tĩnh) | `wo_fft`, `wo_conv`, knock-out |
| H3 | Attention không gian tốt hơn lấy trung bình các kênh | `wo_block2`, `freq_mean` |
| H4 | Bias vị trí tương đối có dấu tốt hơn mã hóa vị trí tuyệt đối/đối xứng, nhất là với các hành động ngược chiều thời gian | `abs_pos` |
| H5 | Mô hình hóa diễn biến giữa các đoạn là cần thiết | `wo_block3` |

## 4. Triển khai và thiết lập thí nghiệm

### 4.1 Dữ liệu

| | SSHAR-ESP | UT-HAR |
|---|---|---|
| Thiết bị | 3 RX ESP32, phòng 2 | Intel 5300, 3 anten |
| Kênh | R = 3, S = 56 | R = 3, S = 30 |
| Độ dài | T = 1000 (5 s, fs = 200 Hz) | T = 250 |
| Lớp | 8: ngồi xuống, đứng lên, cúi nhặt, ngã, nằm im, đi, chạy, phòng trống | 7: nằm xuống, ngã, đi, nhặt, chạy, ngồi xuống, đứng lên |
| Split | theo lần lặp, 6:2:2 (3840 / 1280 / 1280) | split chính thức (3977 / 496 / 500) |

### 4.2 Cấu hình mô hình

| | Giá trị |
|---|---|
| W, bước, K | 64, 32, 30 (ESP); 32, 16, 14 (UT-HAR) |
| D, số head | 64, 4 |
| Số lớp | L₂ = 2, L₃ = 6 |
| FFN | 4D |
| Dropout | 0.1 |
| Quy mô (ESP) | 449K tham số, 2.41 G MACs |
| Chuẩn hóa trên UT-HAR | không chuẩn hóa trong mô hình (giữ min-max của loader), vì trên dữ liệu này mức tĩnh là thông tin quyết định (mục 5.5) |

### 4.3 Huấn luyện

- Adam, lr 1e-3, batch 32, warmup tuyến tính 5 epoch, cosine về 0 ở epoch 100, không dừng sớm.
- Checkpoint theo độ chính xác val cao nhất. Seed 42, một lần chạy.
- GPU Kaggle: khoảng 3.8 giờ trên ESP, 45 phút trên UT-HAR.

### 4.4 Baseline

CNN-GRU (2.0M tham số, 6.70 G MACs) và BiMamba (653K, 0.10 G):
- cùng split và cùng tập test với WiFormer;
- z-score toàn cục theo thống kê tập train, lr 1e-3 cố định, batch 64.

### 4.5 Thước đo

Độ chính xác và F1 macro, tính ở ba mức từ cùng một ma trận nhầm lẫn, không train lại:
- **8 lớp:** đầy đủ.
- **7 lớp:** gộp nằm im và phòng trống thành lớp "tĩnh".
- **6 hành động:** chỉ tính các mẫu thuộc 6 lớp hành động.

Mức 7 và 6 tách khả năng nhận dạng hành động khỏi bài toán phân biệt hai trạng thái tĩnh.

### 4.6 Các biến thể ablation

Mỗi biến thể thay đổi một thành phần, phần còn lại giữ nguyên như mô hình đầy đủ.

| Biến thể | Thay đổi |
|---|---|
| `wo_fft` | Token chỉ có `v_time` |
| `wo_conv` | Token chỉ có `v_freq` |
| `wo_block1` | Token = `Linear(W → D)` trên đoạn thô |
| `abs_pos` | Khối 2: embedding subcarrier tuyệt đối cộng vào token, thay cho bias tương đối. Khối 3: ALiBi đối xứng `−m_h·\|i − j\|`, thay cho bias có dấu |
| `wo_block2` | Khối 2 = trung bình R·S token của đoạn |
| `freq_mean` | Như `wo_conv`, nhưng Khối 2 = trung bình. Tương đương phổ trung bình qua (rx, s) |
| `wo_block3` | Khối 3 = trung bình qua các đoạn |

## 5. Kết quả và nhận xét

### 5.1 So sánh với baseline (SSHAR-ESP)

| Mô hình | Acc 8 | F1 8 | Acc 7 | F1 7 | Acc 6 | F1 6 | Tham số | MACs |
|---|---|---|---|---|---|---|---|---|
| CNN-GRU | 94.92 | 94.91 | 94.92 | 94.20 | 93.23 | 93.31 | 2.0M | 6.70 G |
| BiMamba | 95.31 | 95.31 | 95.31 | 94.68 | 93.85 | 93.95 | 653K | 0.10 G |
| **WiFormer** | **96.80** | **96.79** | **97.97** | **97.68** | **97.29** | **97.29** | 449K | 2.41 G |

Số mẫu nhầm theo cặp lớp (tính cả hai chiều):

| | ngồi ↔ đứng | nội bộ ngồi xuống / đứng lên / cúi nhặt | đi ↔ chạy | nằm im ↔ phòng trống | Tổng lỗi (1280 mẫu) |
|---|---|---|---|---|---|
| CNN-GRU | 12 | 13 | 39 | 0 | 65 |
| BiMamba | 11 | 13 | 37 | 0 | 60 |
| **WiFormer** | **6** | **8** | **16** | 15 | **41** |

**Nhận xét**
- WiFormer giảm khoảng một nửa số lỗi ở các cặp hành động khó (ngồi ↔ đứng, đi ↔ chạy).
- Baseline chỉ tốt hơn ở cặp nằm im ↔ phòng trống, nhờ dùng mức tĩnh qua z-score toàn cục. Đây đúng là thông tin WiFormer cố ý bỏ khi chuẩn hóa từng mẫu.
- Vì vậy khoảng cách nới rộng từ 1.5–1.9 điểm (8 lớp) lên 3.4–4.1 điểm (6 hành động).

### 5.2 Ablation (SSHAR-ESP)

| Biến thể | Acc 8 | Δ | Acc 7 | Δ | Acc 6 | Δ | Tham số | MACs |
|---|---|---|---|---|---|---|---|---|
| **WiFormer (đầy đủ)** | **96.80** | — | **97.97** | — | **97.29** | — | 449K | 2.41 G |
| *Khối 1* | | | | | | | | |
| wo_fft | 95.86 | −0.94 | 97.19 | −0.78 | 96.25 | −1.04 | 443K | 2.38 G |
| wo_conv | 95.16 | −1.64 | 97.27 | −0.70 | 96.35 | −0.94 | 407K | 0.54 G |
| wo_block1 | 91.72 | **−5.08** | 96.02 | **−1.95** | 94.69 | **−2.60** | 405K | 0.53 G |
| *Khối 2* | | | | | | | | |
| wo_block2 | 94.61 | −2.19 | 97.42 | −0.55 | 96.56 | −0.73 | 349K | 1.91 G |
| freq_mean | (bổ sung) | | | | | | 307K | 0.04 G |
| *Mã hóa vị trí* | | | | | | | | |
| abs_pos | 95.62 | −1.18 | 96.64 | −1.33 | 95.52 | −1.77 | 453K | 2.24 G |
| *Khối 3* | | | | | | | | |
| wo_block3 | (bổ sung) | | | | | | 149K | 2.40 G |

Số mẫu nhầm theo cặp lớp:

| | ngồi ↔ đứng | nội bộ ngồi/đứng/cúi | đi ↔ chạy | nằm im ↔ phòng trống | Tổng |
|---|---|---|---|---|---|
| WiFormer | 6 | 8 | 16 | 15 | 41 |
| wo_fft | 10 | 12 | 22 | 17 | 53 |
| wo_conv | 13 | 17 | 16 | 27 | 62 |
| wo_block1 | 14 | 19 | 24 | 55 | 106 |
| wo_block2 | 11 | 14 | 18 | 36 | 69 |
| abs_pos | 12 | 21 | 20 | 13 | 56 |

### 5.3 Mô hình dựa vào góc nhìn nào

Đặt một nửa đầu vào của lớp gộp bằng 0 lúc test (knock-out), không train lại. Chuẩn trọng số gộp của hai nửa gần bằng nhau: thời gian 7.05, phổ 7.66.

| Recall (%) | ngồi xuống | đứng lên | cúi nhặt | ngã | nằm im | đi | chạy | phòng trống | Acc |
|---|---|---|---|---|---|---|---|---|---|
| Cả hai | 96.9 | 98.1 | 98.8 | 100 | 91.9 | 98.1 | 91.9 | 98.8 | 96.8 |
| Chỉ phổ | 60.6 | 100 | 83.8 | 57.5 | 0.0 | 98.8 | 68.8 | 100 | 71.2 |
| Chỉ thời gian | 45.0 | 75.6 | 70.0 | 93.1 | 43.1 | 48.8 | 97.5 | 28.8 | 62.7 |

### 5.4 Đánh giá các giả thuyết

**H1: Khối 1 là cần thiết. Được xác nhận.**
- `wo_block1` tụt nhiều nhất ở cả ba mức.
- Một nửa số lỗi của nó (55/106) nằm ở cặp nằm im ↔ phòng trống. Phân biệt cặp này cần năng lượng của các dao động nhỏ, mà phép chiếu tuyến tính không trích được.
- Biến thể này cũng overfit rõ: val loss thấp nhất là 0.271, so với 0.089 của mô hình đầy đủ.

**H2: Hai góc nhìn bổ sung nhau. Được xác nhận.**
- Mỗi góc nhìn đứng riêng đều kém mô hình đầy đủ, ở cả ba mức.
- Kiểu lỗi khớp với giả thuyết:
  - bỏ phổ làm đi ↔ chạy tăng từ 16 lên 22;
  - bỏ dạng sóng làm nằm im ↔ phòng trống tăng từ 15 lên 27, và nội bộ ngồi/đứng/cúi từ 8 lên 17.
- Knock-out cho cùng bức tranh. Chỉ dùng phổ thì mất hẳn lớp nằm im (0%). Chỉ dùng thời gian thì lớp đi còn 48.8% và lớp phòng trống còn 28.8%.
- Mô hình dùng cả hai góc nhìn với mức độ tương đương: chuẩn trọng số gần bằng nhau, và bỏ bên nào cũng mất 25–34 điểm.

**H3: Attention không gian tốt hơn trung bình. Xác nhận một phần.**
- `wo_block2` tụt 2.19 điểm ở 8 lớp, nhưng chỉ 0.55–0.73 điểm ở 7 lớp và 6 hành động. Lỗi tăng chủ yếu ở nằm im ↔ phòng trống (15 lên 36).
- Trên split theo lần lặp, đóng góp chính của Khối 2 là **chọn kênh mang vết của các dao động nhỏ**, vốn chỉ hiện ở một số subcarrier và RX.
- Với riêng góc nhìn phổ, câu hỏi "attention trên phổ từng subcarrier hay lấy trung bình phổ" được trả lời bằng `freq_mean` so với `wo_conv` (bổ sung).

**H4: Bias tương đối có dấu. Được xác nhận.**
- `abs_pos` giữ nguyên Khối 1, chuẩn hóa, quy mô và cách huấn luyện, chỉ đổi cơ chế mã hóa vị trí. Nó tụt 1.18 điểm ở 8 lớp.
- Mức tụt **tăng** khi bỏ cặp tĩnh (−1.33 ở 7 lớp, −1.77 ở 6 hành động). Lợi ích nằm ở chính các hành động.
- Lỗi tăng mạnh nhất ở nội bộ ngồi xuống / đứng lên / cúi nhặt (8 lên 21), trong đó ngồi ↔ đứng tăng từ 6 lên 12. Đây đúng là các hành động gần như đối xứng gương theo thời gian, mà bias đối xứng không phân biệt được.
- Chưa tách được phần đóng góp của Khối 2 và Khối 3 trong biến thể này.

**H5: Khối 3.** (bổ sung)

**Thứ tự tác động trên 6 hành động:**

| Thành phần bị bỏ | Δ (6 hành động) |
|---|---|
| Khối 1 | −2.60 |
| Mã hóa vị trí tương đối | −1.77 |
| Góc nhìn thời gian | −1.04 |
| Góc nhìn phổ | −0.94 |
| Khối 2 | −0.73 |

### 5.5 UT-HAR

| Mô hình | Acc | F1 |
|---|---|---|
| WiFormer, chuẩn hóa từng mẫu | 71.2 | 58.8 |
| WiFormer, không chuẩn hóa trong mô hình | 93.4 | 90.6 |
| AMCS-TCN (5 seed, có trọng số lớp) | 97.36 ± 0.34 | 95.88 |

**Nhận xét**
- Trên UT-HAR, bỏ mức tĩnh làm mất 22 điểm. Mức tĩnh ở đây có thể mang thông tin về lớp (ví dụ tư thế cuối của nằm xuống, ngồi xuống), không chỉ là dấu vân tay môi trường. Lựa chọn chuẩn hóa vì vậy phụ thuộc dữ liệu.
- Ở cấu hình phù hợp, WiFormer vẫn kém AMCS-TCN khoảng 4 điểm (WiFormer chỉ 1 seed, không dùng trọng số lớp).

### 5.6 Hạn chế

1. **Chỉ có split theo lần lặp.** Val gần bão hòa (97–98%), nên chênh lệch giữa các biến thể nhỏ. Lợi ích chính của chuẩn hóa từng mẫu và của Khối 2 được kỳ vọng ở **cross-user**, nhưng chưa được đánh giá. Cần kiểm định chéo 5 fold theo người, mỗi người được test đúng một lần.
2. **Một seed.** Các chênh lệch dưới 1 điểm (khoảng 10 mẫu) cần thêm seed để khẳng định.
3. **`abs_pos` đổi đồng thời Khối 2 và Khối 3,** nên chưa tách được đóng góp của mã hóa vị trí ở từng khối.
4. **Nằm im ↔ phòng trống** là nguồn lỗi lớn nhất còn lại (15/41).
   - Phân biệt có nguyên tắc cần vi chuyển động như nhịp thở (0.2–0.5 Hz).
   - Độ phân giải phổ của một đoạn (fs/W = 3.125 Hz) không thấy được nhịp này; chỉ Khối 3 có thể học qua biến thiên giữa các đoạn.

## 6. Kết luận

WiFormer mô hình hóa CSI theo đúng ba thang của nó: trong đoạn, giữa các kênh, và theo thời gian.
- **Token gộp sớm hai góc nhìn dạng sóng và phổ** cho phép hai nguồn thông tin bổ sung nhau tương tác ở mọi tầng.
- **Bias vị trí tương đối** giúp mô hình không phụ thuộc danh tính RX, và phân biệt được chiều thời gian.

Trên SSHAR-ESP, WiFormer đạt 96.80% (97.29% trên 6 hành động), vượt CNN-GRU và BiMamba với ít tham số hơn. Ablation xác nhận phần lớn giả thuyết thiết kế, và cho thấy mỗi thành phần xử lý một nhóm lỗi riêng:

| Nhóm lỗi | Thành phần chịu trách nhiệm |
|---|---|
| đi ↔ chạy | góc nhìn phổ |
| ngồi ↔ đứng | bias có dấu, góc nhìn dạng sóng |
| nằm im ↔ phòng trống | góc nhìn dạng sóng, Khối 2 |

Hướng tiếp theo:
1. Đánh giá cross-user bằng 5 fold theo người, và thêm seed.
2. Hoàn thiện ablation Khối 3 và `freq_mean`.
3. Cải thiện góc nhìn phổ: cửa sổ Hann, nhúng phi tuyến, kèm phổ trung bình làm ngữ cảnh.
4. Khai thác vi chuyển động để phân biệt các trạng thái tĩnh mà không cần mức tĩnh.
