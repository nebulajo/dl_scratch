# numpy 
import numpy as np

# A shape : (3, 2)
A = [
    [3, 2]
    [1, 2]
    [4, 2]
]

# B shape : (2, 4)
B = [
    [1, 4, 1, 5],
    [1, 4, 1, 5],
]

np.dot(A, B)
np.matmul(A, B)

# for문은 3개이고 얼만큼 반복했는지가 핵심이다.
def mat_mul(A, B):
    
    row_a = len(A) ; col_a = len(A[0]) 
    row_b = len(B) ; col_b = len(B[0])

    C = [[0]*col_b for _ in range(row_a)]

    for i in range(row_a): # 첫번째 행렬의 행
        for l in range(col_b): # 두번째 행렬의 열
            value = 0
            for j in range(col_a): # 첫번째, 두번째 행렬의 동일한 중간 차원 값
                value += A[i][j]*B[j][l]
            C[i][l] = value
            
    print(C)
    
    return 

