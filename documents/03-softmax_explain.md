# Giải thích về Softmax trong Neural Network

#### Mục 1: Tại sao chúng ta cần Softmax? Vấn đề là gì?

Ở các chương trước, bạn có thể đã làm quen với hàm kích hoạt **ReLU (Rectified Linear Unit)**. ReLU rất tốt cho các lớp ẩn (hidden layers) nhưng lại có một vài vấn đề nếu dùng cho lớp cuối cùng (output layer) của một mạng phân loại:

1.  **Không bị chặn (Unbounded):** Đầu ra của ReLU có thể là bất kỳ số dương nào (ví dụ: `[12, 99, 318]`). Những con số này đứng một mình không có nhiều ý nghĩa. 318 lớn hơn 99, nhưng lớn hơn "bao nhiêu"? Liệu nó có "chắc chắn" hơn nhiều không? Chúng ta không có một "ngữ cảnh" để so sánh.
2.  **Không được chuẩn hóa (Not Normalized):** Các giá trị đầu ra không có mối liên hệ tổng thể. Tổng của chúng không bằng một con số cố định nào cả.
3.  **Độc quyền (Exclusive):** Đầu ra của mỗi neuron là độc lập với các neuron khác.

**Mục tiêu của chúng ta:** Với bài toán phân loại, chúng ta muốn mạng nơ-ron "nói" cho chúng ta biết nó "nghĩ" rằng đầu vào thuộc về lớp nào với một mức độ tự tin (confidence) rõ ràng. Ví dụ, với 3 lớp (chó, mèo, chim), chúng ta muốn đầu ra có dạng như `[0.05, 0.9, 0.05]`, nghĩa là: "Tôi tin chắc 90% đây là mèo, 5% là chó và 5% là chim."

=> **Softmax ra đời để giải quyết vấn đề này.** Nó nhận vào các số thực bất kỳ (có thể âm, dương, lớn, nhỏ) và biến chúng thành một **phân phối xác suất (probability distribution)**. Các đặc điểm của phân phối xác suất này là:
*   Tất cả các giá trị đầu ra đều nằm trong khoảng `[0, 1]`.
*   Tổng của tất cả các giá trị đầu ra **luôn bằng 1**.

Những giá trị này chính là **điểm tự tin (confidence scores)** mà chúng ta cần.

#### Mục 2: "Giải phẫu" công thức Softmax

Công thức trong sách có vẻ đáng sợ:
$$ S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}} $$

Đừng lo, hãy chia nó thành 2 bước cực kỳ đơn giản:

**Bước 1: Lũy thừa hóa (Exponentiation) - Tử số `e^z`**

*   `z` là các giá trị đầu ra từ lớp trước (ví dụ `layer_outputs = [4.8, 1.21, 2.385]`).
*   `e` là hằng số Euler (xấp xỉ 2.71828), là cơ số của logarit tự nhiên.
*   "Lũy thừa hóa" đơn giản là lấy `e` mũ các giá trị `z` đó. Trong Python, chúng ta dùng `E ** output` hoặc `math.exp(output)`.

```python
# Ví dụ từ sách
layer_outputs = [4.8, 1.21, 2.385]
E = 2.71828182846

# Tính e^z cho mỗi giá trị
exp_values = [E**4.8, E**1.21, E**2.385] 
# Kết quả: [121.51, 3.35, 10.86]
```

**Tại sao phải làm bước này?**
1.  **Loại bỏ số âm:** `e` mũ bất cứ số nào cũng luôn cho ra kết quả **dương**. Điều này rất quan trọng vì xác suất không thể là số âm.
2.  **Khuếch đại sự khác biệt:** Hàm mũ làm cho các giá trị lớn càng lớn hơn một cách vượt trội so với các giá trị nhỏ. Giá trị `4.8` chỉ lớn hơn `2.385` khoảng 2 lần, nhưng sau khi lũy thừa, `121.51` lớn hơn `10.86` tới hơn 11 lần! Điều này giúp mạng "tự tin" hơn vào dự đoán có điểm số cao nhất.

**Bước 2: Chuẩn hóa (Normalization) - Phép chia**

Sau khi có các giá trị đã được lũy thừa (`exp_values`), chúng ta chỉ cần làm một việc:
1.  Tính tổng tất cả các giá trị đó (mẫu số $\sum_{l=1}^{L} e^{z_{i,l}}$).
2.  Lấy từng giá trị chia cho tổng vừa tính được.

```python
# Tiếp nối ví dụ trên
exp_values = [121.51, 3.35, 10.86]

# 1. Tính tổng
norm_base = sum(exp_values) # 121.51 + 3.35 + 10.86 = 135.72

# 2. Chia từng giá trị cho tổng
norm_values = [
    121.51 / norm_base, # ~0.895
    3.35 / norm_base,   # ~0.025
    10.86 / norm_base   # ~0.080
]

# Kết quả: [0.895, 0.025, 0.080]
# Kiểm tra: 0.895 + 0.025 + 0.080 = 1.0
```

Vậy là xong! Chúng ta đã biến `[4.8, 1.21, 2.385]` thành một phân phối xác suất `[0.895, 0.025, 0.080]`.

#### Mục 3: Tối ưu với NumPy và xử lý theo Lô (Batch)

Trong thực tế, chúng ta không xử lý từng mẫu dữ liệu một mà xử lý cả một **lô (batch)** để tăng tốc độ. Một lô dữ liệu sẽ có dạng một ma trận, trong đó mỗi hàng là đầu ra cho một mẫu.

```python
# Một lô có 3 mẫu, mỗi mẫu có 3 đầu ra
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])
```

Bây giờ, chúng ta cần tính Softmax cho **từng hàng một**. Đây là lúc các tham số `axis` và `keepdims` của NumPy phát huy tác dụng.

*   `np.exp(layer_outputs)`: NumPy thông minh sẽ tự động tính lũy thừa cho mọi phần tử trong ma trận.
*   `np.sum(..., axis=1)`: Chúng ta cần tính tổng của các giá trị **trên mỗi hàng**.
    *   `axis=0`: tính tổng theo cột.
    *   `axis=1`: tính tổng theo hàng. Đây là cái chúng ta cần.
*   `keepdims=True`: Khi tính tổng theo `axis=1`, kết quả sẽ là một vector hàng `[8.395, 7.29, 2.487]`. Nếu chúng ta lấy ma trận `(3, 3)` chia cho vector `(3,)`, NumPy có thể báo lỗi hoặc không thực hiện đúng phép chia theo hàng. `keepdims=True` sẽ giữ nguyên số chiều, biến kết quả thành một vector cột `[[8.395], [7.29], [2.487]]` có shape `(3, 1)`. Lúc này, NumPy có thể thực hiện phép chia ma trận `(3, 3)` cho vector cột `(3, 1)` một cách chính xác (mỗi hàng của ma trận được chia cho giá trị tương ứng trong vector cột).

#### Mục 4: "Bí kíp" chống tràn số (Overflow Prevention)

Hàm mũ `e^x` tăng rất nhanh. Nếu đầu vào `z` là một số lớn (ví dụ `1000`), `np.exp(1000)` sẽ trả về `inf` (vô cực), gây ra lỗi tràn số (overflow) và làm hỏng toàn bộ phép tính.

**Giải pháp:** Chúng ta có thể trừ đi một số bất kỳ từ tất cả các giá trị đầu vào `z` mà không làm thay đổi kết quả cuối cùng của Softmax. Tại sao? Vì tính chất của phép lũy thừa và phép chia:
$$ \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{e^{z_1} \cdot e^{-C}}{e^{z_1} \cdot e^{-C} + e^{z_2} \cdot e^{-C}} = \frac{e^{z_1 - C}}{e^{z_1 - C} + e^{z_2 - C}} $$

Vậy chúng ta nên trừ đi số nào? **Số lớn nhất (max)** trong các giá trị đầu vào của hàng đó.

```python
inputs = [1, 2, 3]
max_value = 3
shifted_inputs = [1-3, 2-3, 3-3] # -> [-2, -1, 0]
```

Lợi ích của việc này:
1.  Giá trị lớn nhất sau khi trừ sẽ là `0`. (`e^0 = 1`)
2.  Tất cả các giá trị khác sẽ là số âm. (`e` mũ số âm luôn là một số nhỏ hơn 1).
3.  Điều này đảm bảo rằng đầu vào cho hàm `exp` sẽ không bao giờ là một số dương lớn, từ đó **ngăn chặn hoàn toàn lỗi tràn số**.

Đây chính là lý do trong đoạn code cuối cùng của sách, bạn sẽ thấy dòng này:
```python
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
```
Đây là phiên bản Softmax hoàn chỉnh, an toàn và hiệu quả.

***

### Phần 2: Giải Thích Trừu Tượng & Dễ Hiểu

Hãy quên đi công thức toán học và code. Hãy tưởng tượng Softmax là một **"Máy Phân Bổ Sự Tự Tin"** cho một cuộc thi tài năng.

**1. Vòng Sơ Khảo - Điểm thô (Raw Scores)**

Giả sử có 3 thí sinh (Chó, Mèo, Chim) tham gia một cuộc thi. Ban giám khảo (lớp mạng nơ-ron phía trước) cho điểm thô. Điểm này có thể rất lộn xộn:
`Điểm thô = [4.8, 1.21, 2.385]`

Nhìn vào điểm này, chúng ta biết thí sinh Chó có điểm cao nhất, nhưng "cao hơn" như thế nào? Mức độ "chiến thắng" có áp đảo không? Rất khó nói.

**2. Bước 1: Máy "Hype" - Lũy thừa hóa**

Để làm cho kết quả rõ ràng hơn, MC cho các điểm số này vào một cái **Máy "Hype"**. Máy này có 2 chức năng:
*   **Không có điểm âm:** Nó biến mọi điểm số thành điểm "nhiệt tình" (luôn dương).
*   **Tâng bốc người giỏi nhất:** Máy này cực kỳ "thiên vị". Ai điểm cao sẵn rồi sẽ được tâng bốc lên tận mây xanh, trong khi người điểm thấp chỉ được tăng nhẹ.

Sau khi qua Máy "Hype" (tức là `e^x`):
`Điểm Hype = [121.5, 3.4, 10.9]`

Bây giờ thì sự khác biệt đã quá rõ ràng! Thí sinh Chó không chỉ cao điểm hơn, mà còn **áp đảo hoàn toàn** phần còn lại.

**3. Bước 2: Chia "Chiếc Bánh Tự Tin" - Chuẩn hóa**

Bây giờ, để khán giả dễ hiểu, MC quyết định không dùng điểm hype nữa mà sẽ chia một "chiếc bánh tự tin" 100% cho 3 thí sinh, dựa trên tỷ lệ điểm hype của họ.

*   Tổng điểm Hype = 121.5 + 3.4 + 10.9 = 135.8
*   Phần bánh của Chó: `121.5 / 135.8 ≈ 89.5%`
*   Phần bánh của Mèo: `3.4 / 135.8 ≈ 2.5%`
*   Phần bánh của Chim: `10.9 / 135.8 ≈ 8.0%`

**Kết quả cuối cùng:**
`Mức độ tự tin = [0.895, 0.025, 0.080]`

Đây chính là đầu ra của Softmax. Nó cho chúng ta một kết luận rất rõ ràng: "Dựa trên màn trình diễn, tôi **tin chắc 89.5%** người chiến thắng là Chó."

**Về "bí kíp" chống tràn số:** Hãy tưởng tượng một giám khảo quá phấn khích cho điểm `1000`. Máy "Hype" sẽ bị "cháy" (overflow). MC thông minh nhận ra rằng điều quan trọng là **sự chênh lệch điểm số** chứ không phải bản thân điểm số. Vì vậy, trước khi cho vào máy, anh ta tìm điểm cao nhất (1000) và trừ nó khỏi điểm của mọi người. Kết quả cuối cùng sau khi chia bánh vẫn không hề thay đổi, nhưng cái máy hype đã được cứu!

**Tóm lại, Softmax làm hai việc:**
1.  **Sử dụng hàm mũ `e^x` để khuếch đại điểm số cao nhất, biến nó thành "người dẫn đầu" rõ rệt.**
2.  **Chuẩn hóa các điểm số đã được khuếch đại đó thành một tỷ lệ phần trăm (hoặc xác suất), để tất cả cộng lại bằng 1.**