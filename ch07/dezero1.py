import numpy as np
from dezero import Variable
import dezero.functions as F

# ベクトルの内積
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(a), Variable(b)  # 省略可能
c = F.matmul(a, b)
print(c)

# 行列の積
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = F.matmul(a, b)
print(c)