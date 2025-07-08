# Đi sâu vào "phòng máy" của mạng nơ-ron và xem xét từng phép nhân, phép cộng ma trận

Giả sử chúng ta có một bài toán đơn giản:
*   **Đầu vào:** Một vector có 2 đặc trưng (ví dụ: chiều cao, cân nặng).
*   **Kiến trúc mạng:**
    *   Lớp ẩn (Hidden Layer) có 3 neuron.
    *   Lớp ra (Output Layer) có 2 neuron (tương ứng 2 lớp, ví dụ: "Loại A" và "Loại B").
*   **Hàm kích hoạt:** ReLU cho lớp ẩn, Softmax cho lớp ra.

Hãy bắt đầu với các giá trị cụ thể.

---

### Sơ Đồ Chi Tiết: Từng Bước Tính Toán Ma Trận

**Bước 0: Khởi tạo các tham số**

*   **Dữ liệu đầu vào (một mẫu):**
    `inputs = [1, 2]`  (shape: 1x2)

*   **Trọng số và bias Lớp Ẩn (W1, b1):**
    `W1 = [[0.2, 0.8, -0.5],`
    `      [0.5, -0.9, 0.3]]` (shape: 2x3)

    `b1 = [2, 3, 0.5]` (shape: 1x3)

*   **Trọng số và bias Lớp Ra (W2, b2):**
    `W2 = [[0.1, -0.4],`
    `      [-0.2, 0.6],`
    `      [0.7, -0.9]]` (shape: 3x2)

    `b2 = [0.1, -0.2]` (shape: 1x2)

---

**Bước 1: Tính toán tại Lớp Ẩn (Hidden Layer)**

**Phép toán:** `z1 = inputs · W1 + b1` (Phép nhân ma trận `·` và cộng vector)

```
        [ Trọng số W1 ]
         (shape 2x3)
      [[0.2, 0.8, -0.5],
       [0.5, -0.9, 0.3]]
            ^
            |
[inputs] ·--+
(1x2)
[1, 2]

================== Phép nhân ma trận (inputs · W1) ==================

Kết quả (1x3) = [(1*0.2 + 2*0.5), (1*0.8 + 2*-0.9), (1*-0.5 + 2*0.3)]
              = [(0.2 + 1.0),   (0.8 - 1.8),    (-0.5 + 0.6)  ]
              = [1.2,           -1.0,           0.1           ]

================== Phép cộng bias ( + b1 ) =======================

  [1.2, -1.0, 0.1]
+ [2.0,  3.0, 0.5]
--------------------
= [3.2,  2.0, 0.6]  <== Đây là z1 (logits của lớp ẩn)
```

**Bước 2: Áp dụng hàm kích hoạt ReLU**

**Phép toán:** `h1 = ReLU(z1) = max(0, z1)`

```
  Input vào ReLU: z1 = [3.2, 2.0, 0.6]

  max(0, 3.2) -> 3.2
  max(0, 2.0) -> 2.0
  max(0, 0.6) -> 0.6

  Kết quả: h1 = [3.2, 2.0, 0.6] <== Đầu ra của lớp ẩn sau kích hoạt
  (Trong trường hợp này không có giá trị nào bị cắt vì tất cả đều dương)
```

---

**Bước 3: Tính toán tại Lớp Ra (Output Layer)**

**Phép toán:** `z_out = h1 · W2 + b2`

```
        [ Trọng số W2 ]
         (shape 3x2)
      [[0.1, -0.4],
       [-0.2, 0.6],
       [0.7, -0.9]]
            ^
            |
   [h1] ·---+
  (1x3)
[3.2, 2.0, 0.6]

================== Phép nhân ma trận (h1 · W2) =====================

Kết quả (1x2) = [ (3.2*0.1 + 2.0*-0.2 + 0.6*0.7),  (3.2*-0.4 + 2.0*0.6 + 0.6*-0.9) ]
              = [ (0.32  - 0.4    + 0.42),       (-1.28    + 1.2    - 0.54)       ]
              = [ 0.34,                          -0.62                         ]

================== Phép cộng bias ( + b2 ) ========================

  [0.34, -0.62]
+ [0.1,  -0.2]
--------------------
= [0.44, -0.82] <== Đây là z_out (logits cuối cùng trước Softmax)
```

---

**Bước 4: Áp dụng hàm kích hoạt Softmax**

**Phép toán:** `probabilities = Softmax(z_out)`

1.  **Lũy thừa hóa:** `e^z_out`
    `e^0.44  ≈ 1.55`
    `e^-0.82 ≈ 0.44`

2.  **Tính tổng:**
    `Tổng = 1.55 + 0.44 = 1.99`

3.  **Chuẩn hóa:**
    `Xác suất Lớp A = 1.55 / 1.99 ≈ 0.779`
    `Xác suất Lớp B = 0.44 / 1.99 ≈ 0.221`

---

### KẾT QUẢ CUỐI CÙNG

```
  Input: [1, 2]
    |
    V
  z1 = [3.2, 2.0, 0.6]  (Sau Lớp Ẩn 1)
    |
    V (ReLU)
  h1 = [3.2, 2.0, 0.6]
    |
    V
  z_out = [0.44, -0.82] (Sau Lớp Ra)
    |
    V (Softmax)
  Probabilities = [0.779, 0.221]

==> Dự đoán: "Loại A" với xác suất 77.9%
```

Sơ đồ chi tiết này cho thấy chính xác cách một vector đầu vào `[1, 2]` đi qua từng lớp, qua từng phép nhân ma trận, cộng bias và hàm kích hoạt để cuối cùng cho ra một dự đoán xác suất cụ thể. Mỗi bước đều là một phép toán ma trận hoặc vector đơn giản. Quá trình "học" của mạng nơ-ron chính là việc điều chỉnh các giá trị trong ma trận `W1, b1, W2, b2` để kết quả cuối cùng ngày càng chính xác.