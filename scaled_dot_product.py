import numpy as np

def softmax(x, axis=-1):

    # 오버플로 방지를 위해 x_max로 빼주기 - 지수 함수여서 값이 너무 커진다.(컴퓨터가 표현하기에)
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max) # 최대값을 뺀 평균 값은 동일하다.  - 최종적으로 0~1사이의 확률값 나온다.
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product(q, k, v):
    
    print(f"q shape : {q.shape}")
    print(f"k shape : {k.shape}")
    print(f"v shape : {v.shape}")

    d_k = len(k[0])
    sqrt_d_k = np.sqrt(d_k)

    # query, key 내적    
    qk = np.matmul(q, k.T)
    qk_sqrt_d_k = qk/sqrt_d_k # sqrt d_k로 scaling한다. -> attention score
    
    # soft max 적용 - attention score
    att_scores = softmax(qk_sqrt_d_k)     
        
    return np.matmul(att_scores, v)



if __name__ == "__main__":

    # diffusion에서 image 생성 모델이며 text가 condition인 모델인 경우를 고려한다.
    # 이때 텍스트에서 단어의 개수는 3개 이며, 이미지 패티치의 개수는 4개인 상황의 scaled-dot-product를 구한다.

    n_text = 3 ; n_patch = 4
    
    # condition값(text)은 query로 둔다.
    seq_q = n_text
    d_q = 6 # 데이터 차원은 임의로 지정한다. - 데이터 차원은 모두 동일하다. 
    
    # image는 key, value로 둔다. - 주어진 condition(text, query)에 key, value로 지정한 image와의 관계를 계산
    seq_k = n_patch
    d_k = d_q # key의 데이터 차원은 query의 차원과 동일하다. (내적 수행해야 하기 때문에)
    # cross-attention 구현할 때는 동일한 데이터 차원으로 nn.Linear를 통해 projection 한 Q, K, V를 사용한다.
    
    seq_v, d_v = seq_k, d_k # key, value는 동일한 데이터에 대한 값이므로 데이터 개수 및 차원이 모두 동일하다.
    
        
    # Query, Key, Value 랜덤 초기화
    q = np.random.rand(seq_q, d_q)
    k = np.random.rand(seq_k, d_k)
    v = np.random.rand(seq_v, d_v)
    
    print(scaled_dot_product(q, k, v))