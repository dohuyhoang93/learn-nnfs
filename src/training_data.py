import numpy as np
import nnfs
import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data
nnfs.init()
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()


# Dữ liệu mẫu
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Tạo biểu đồ đường
plt.plot(x, y, label='Dữ liệu mẫu', color='blue', marker='o')

# Thêm tiêu đề và nhãn
plt.title('Biểu đồ Đường Đơn Giản')
plt.xlabel('Trục X')
plt.ylabel('Trục Y')

# Hiển thị chú thích
plt.legend()

# Hiển thị biểu đồ
plt.show()