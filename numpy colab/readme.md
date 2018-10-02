使用Google Colab完成矩陣相乘的作業

```
def matrixMul(A, B):
    res = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    return res

def matrixMul2(A, B):
    return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]

a = [[1,2], [3,4], [5,6], [7,8]]
b = [[1,2,3,4], [5,6,7,8]]

from numpy import dot
print map(list,dot(a,b))
print map(list,dot(b,a))
```
![numpy01](https://github.com/yiling0308/AI_and_security/blob/master/numpy%20colab/picture/1.png)
