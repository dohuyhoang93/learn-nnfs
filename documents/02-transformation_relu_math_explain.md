# Mổ xẻ quá trình biến đổi một điểm dữ liệu duy nhất.

Giả sử chúng ta có:
*   Một điểm dữ liệu đầu vào `O` có tọa độ `(0.5, 1.0)`.
*   Một lớp `Layer_Dense` có 2 đầu vào và 3 neuron.

### 1. Khởi Tạo (Initialization)

Lớp `dense1` được khởi tạo với `weights` và `biases`. Giả sử sau khi khởi tạo ngẫu nhiên, chúng ta có các giá trị cụ thể như sau:

#### Ma trận Trọng số `W` (self.weights) - Kích thước (2, 3)
*   **Hàng 0:** Trọng số cho đầu vào thứ nhất (`i1 = 0.5`).
*   **Hàng 1:** Trọng số cho đầu vào thứ hai (`i2 = 1.0`).
*   **Cột 0, 1, 2:** Tương ứng với Neuron 0, 1, 2.

```
          Neuron 0   Neuron 1   Neuron 2
         +----------+----------+----------+
Input 0  |   0.2    |   0.8    |  -0.5    |
(i1=0.5) +----------+----------+----------+
Input 1  |  -0.9    |   0.2    |   0.4    |
(i2=1.0) +----------+----------+----------+
```

#### Vector Bias `b` (self.biases) - Kích thước (1, 3)
*   Mỗi giá trị tương ứng với thiên kiến của một neuron.

```
         +----------+----------+----------+
         |   2.0    |   3.0    |   0.5    |
         +----------+----------+----------+
           Neuron 0   Neuron 1   Neuron 2
```

### 2. Quá trình Biến đổi (Transformation) - `dense1.forward(O)`

Chúng ta thực hiện phép toán: `v' = v · W + b`

#### Bước 2.1: Phép nhân ma trận (Dot Product) `v · W`

*   `v` là vector đầu vào: `[0.5, 1.0]` (Kích thước 1x2)
*   `W` là ma trận trọng số (Kích thước 2x3)
*   Kết quả sẽ là một vector kích thước 1x3.

```ascii
                                       +-------+-------+-------+
                                       |  0.2  |  0.8  | -0.5  |
                                       | -0.9  |  0.2  |  0.4  |
                                       +-------+-------+-------+
                                                 ^
                                                 |
                                                 · (Dot Product)
+-------+-------+
|  0.5  |  1.0  |
+-------+-------+
      |
      +-------------------------------------------------------------+
      |                                                             |
      v                                                             v
    Tính toán cho Neuron 0:                                       Tính toán cho Neuron 1:
    (0.5 * 0.2) + (1.0 * -0.9)                                    (0.5 * 0.8) + (1.0 * 0.2)
    = 0.1 - 0.9                                                   = 0.4 + 0.2
    = -0.8                                                        = 0.6

                                                                     Tính toán cho Neuron 2:
                                                                     (0.5 * -0.5) + (1.0 * 0.4)
                                                                     = -0.25 + 0.4
                                                                     = 0.15
```

Kết quả của phép nhân ma trận là vector `[-0.8, 0.6, 0.15]`.

#### Bước 2.2: Cộng Vector Bias `+ b`

Bây giờ, chúng ta lấy kết quả ở trên và cộng với vector bias.

```ascii
      Kết quả từ v · W                  Vector Bias b                Vector đầu ra v'
+--------+-------+--------+     +     +-------+-------+-------+     =     +-------+-------+-------+
|  -0.8  |  0.6  |  0.15  |           |  2.0  |  3.0  |  0.5  |           |  1.2  |  3.6  |  0.65 |
+--------+-------+--------+           +-------+-------+-------+           +-------+-------+-------+
     |        |        |                 |        |        |                 |        |        |
     |        |        +-----------------|--------|--------|-----------------+        |
     |        +--------------------------|--------|--------+--------------------------+
     +-----------------------------------|--------+-----------------------------------+

     -0.8 + 2.0 = 1.2
           0.6 + 3.0 = 3.6
                 0.15 + 0.5 = 0.65
```

**Kết quả:** Vector `v'` (đầu ra của `dense1`) là `[1.2, 3.6, 0.65]`.
Đây chính là tọa độ của điểm `O` trong không gian 3 chiều mới sau phép biến đổi tuyến tính.

### 3. Kích Hoạt ReLU - `activation1.forward(v')`

Bây giờ, chúng ta đưa vector `v'` qua hàm ReLU. Hàm này hoạt động trên **từng phần tử (element-wise)**.

```ascii
     Vector đầu vào v' cho ReLU           Hành động của ReLU           Vector cuối cùng v''
+-------+-------+-------+        max(0, x)       +-------+-------+-------+
|  1.2  |  3.6  |  0.65 |  ---------------->     |  1.2  |  3.6  |  0.65 |
+-------+-------+-------+                        +-------+-------+-------+
     |        |        |
     |        |        +-----> max(0, 0.65) = 0.65
     |        +--------------> max(0, 3.6)  = 3.6
     +-----------------------> max(0, 1.2)  = 1.2
```

Trong ví dụ này, vì tất cả các thành phần của `v'` đều là số dương, nên đầu ra của ReLU `v''` giống hệt `v'`.

**Nếu `v'` là `[-0.8, 0.6, 0.15]` (trước khi cộng bias), thì kết quả sẽ khác:**

```ascii
     Vector đầu vào v' cho ReLU           Hành động của ReLU           Vector cuối cùng v''
+--------+-------+--------+        max(0, x)       +-------+-------+--------+
|  -0.8  |  0.6  |  0.15  |  ---------------->     |  0.0  |  0.6  |  0.15  |
+--------+-------+--------+                        +-------+-------+--------+
     |        |        |
     |        |        +-----> max(0, 0.15) = 0.15
     |        +--------------> max(0, 0.6)  = 0.6
     +-----------------------> max(0, -0.8) = 0.0
```

Sơ đồ này đã mô tả toàn bộ quá trình toán học từ một vector đầu vào `v` đến vector cuối cùng `v''` sau khi qua một lớp dày đặc và một lớp kích hoạt ReLU.

# Diễn giải trừu tượng
**Mỗi neuron đóng góp vào việc tạo ra "chữ ký" cuối cùng như thế nào?**

### Phân Tích
1.  **Neural 1 đóng góp một phần** vào việc tạo ra chữ ký cuối cùng.<br>
Nó giống như một nhạc công trong dàn nhạc. Nhạc công violin không "mang" bản giao hưởng, anh ta chỉ chơi phần violin của mình. Bản giao hưởng (chữ ký) là sự kết hợp của tất cả các nhạc công.<br>
**Cách diễn đạt chính xác hơn:** "Neuron 1 có một bộ **tiêu chí riêng** (weights và bias của nó)."


2.  **"biến đổi O(x,y) --> O'(x,y,z)"**: Toàn bộ lớp (gồm cả 3 neuron) cùng nhau thực hiện phép biến đổi này.

### Diễn Giải Lại

1.  **Mỗi Neuron là một "Máy Đo Đặc Trưng"**:
    *   **Neuron 1** được trang bị một bộ tiêu chí `(w1, b1)`. Nó đo xem điểm `O(x,y)` phù hợp với tiêu chí này đến đâu và cho ra một điểm số là `x'`.
    *   **Neuron 2** được trang bị một bộ tiêu chí `(w2, b2)`. Nó đo xem điểm `O(x,y)` phù hợp với tiêu chí này đến đâu và cho ra một điểm số là `y'`.
    *   **Neuron 3** được trang bị một bộ tiêu chí `(w3, b3)`. Nó đo xem điểm `O(x,y)` phù hợp với tiêu chí này đến đâu và cho ra một điểm số là `z'`.

2.  **Tạo Ra "Chữ Ký"**:
    *   **"Chữ ký"** của điểm `O` không phải là do một neuron tạo ra. **"Chữ ký" chính là vector kết quả `O'(x', y', z')`**. Nó là tập hợp các điểm số mà tất cả các "máy đo" đã đưa ra.

3.  **Mục Tiêu Huấn Luyện (Training)**:
    *   Quá trình huấn luyện sẽ **điều chỉnh các bộ tiêu chí `(w, b)` của từng neuron** sao cho:
        *   Tất cả các điểm `O` thuộc lớp "Xanh" khi đi qua 3 "máy đo" này sẽ tạo ra các vector `O'` (các chữ ký) nằm gần nhau trong một vùng không gian.
        *   Tất cả các điểm `O` thuộc lớp "Đỏ" sẽ tạo ra các chữ ký nằm gần nhau trong một vùng không gian **khác**.
        *   Và tương tự cho lớp "Lá".

**Sơ Đồ ASCII Cập Nhật để Phản Ánh Ý Tưởng Này**

```
  Điểm Đầu Vào O(x,y)
          |
          |
+---------+---------+
|                   |
v                   v
Máy Đo 1            Máy Đo 2            Máy Đo 3
(Tiêu chí w1, b1)   (Tiêu chí w2, b2)   (Tiêu chí w3, b3)
|                   |                   |
v                   v                   v
Điểm số x'          Điểm số y'          Điểm số z'
|                   |                   |
+---------+---------+-------------------+
          |
          v
"Chữ Ký" = O'(x', y', z')
(Vector kết quả trong không gian mới)

```

**Ví dụ:** Sau khi huấn luyện, có thể xảy ra trường hợp:
*   **Tiêu chí 1** (của Neuron 1) trở thành "phát hiện đường cong hướng lên".
*   **Tiêu chí 2** (của Neuron 2) trở thành "phát hiện vị trí gần gốc tọa độ".
*   Một điểm `O` thuộc lớp "Xanh" có thể vừa cong lên, vừa gần gốc tọa độ. Chữ ký của nó sẽ là `O'(CAO, CAO, ...)`.
*   Một điểm `O` thuộc lớp "Đỏ" có thể cong lên nhưng xa gốc tọa độ. Chữ ký của nó sẽ là `O'(CAO, THẤP, ...)`.

**Kết luận:** Mỗi neuron có một vai trò riêng. Vai trò đó là **"đo lường một đặc trưng"**. "Chữ ký" cuối cùng của một điểm dữ liệu là **tổ hợp kết quả** từ tất cả các phép đo đặc trưng đó.