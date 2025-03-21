# 행렬 인덱스만 뒤바꿔주면 된다.

def transpose(A):
    row = len(A) ; col = len(A[0])
    answer =[[0]*row for _ in range(col)]
    
    for i in range(row):
        for j in range(col):
            answer[j][i] = A[i][j]
    
    print(answer)
    
    return
