
# 1번 문제

numpy를 사용하여 다음과 행렬을 생성하고, 슬라이싱 기법을 활용하여 [result]와 같은 결과를 출력하는 코드를 작성하시오.

```

[matrix]
[[0 1 2 3 4]
[5 6 7 8 9]
[10 11 12 13 14]]

[result]
7
14
[6 7]
[7 12]
[[3 4]
[8 9]]

```

### 풀이

```python

import numpy as np

def matrixSlicing():
    # 3행 5열의 2차원 배열을 0~14값으로 채움
    arr = np.arange(15).reshape((3,5))
    print(arr[1][2]) # 7
    print(arr[2][4]) # 14
    print(arr[1][1:3]) # [6 7]
    print(arr[1:3,2]) # [7 12]
    print(arr[0:2, 3:5]) # [[3 4] [8 9]]

matrixSlicitn()
```

<br>

# 2번 문제

numpy를 사용하여 다음과 같은 행렬을 생성하고, 산술연산자 및 인덱싱 기법을 활용하여 [result]와 같은 결과를 출력하는 코드를 작성하시오.


```
[matrix]
[ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]

[result]
[ 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]
[ 1 4 7 10 13 16 19 22 25 28] : 0 3 6 9 
[ 4 10 16 22 28] : 3 9 15 21 27 (3*1, 3*3, 3*5, 3*7, 3*9)
```

### 풀이

```python
import numpy as np
    
def calcIndexing():
    arr = np.arange(1,31) # 1부터 30까지의 값을 arr 배열에 넣음
    print(arr[:15]*2) # 15번 인덱스까지 요소의 2배 값을 출력
    print(arr.reshape((10,3))[:,0])
    print(arr.reshape((5,6))[:,3]) # 5x6 모양으로 변경 후 3번째 열의 요소들 출력

calcIndexing()
```