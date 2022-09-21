# 4. Numpy

# NumPy 배열이란?

- `ndarray` : 배열 중심의 빠른 산술 연산을 지원하는 다차원 배열 지원 (행렬 연산)
- 반복되는 루프를 가지지 않고, 전체 배열에서의 빠른 연산을 지원
- 디스크에 배열 데이터를 읽고 쓰기 위한 도구 지원
- 선형 대수, random number 생성, 푸리에 연산 등 고급 수학 연산 지원
- C, C++ 로 작성된 라이브러리를 C API를 사용해 NumPy 와 연동

<br>

## 데이터 분석을 위한 NumPy

- 빠른 벡터화된 배열 연산 제공
    - Data merging, clearning, subsetting & filtering, transformation
- 일반적인 배열 알고리즘 지원
    - sort, unique, set operation
- 효율적인 데이터 통계 및 결합/요약 기능 제공
- 반복 루프를 사용하지 않고 배열의 조건부 제어 기능 제공
- 그룹 단위의 데이터 처리 지원
    - aggregation, transformation, function application

<br>

## ndarray

- 모든 배열 요소는 동일한 자료구조를 가짐
- shape: 행렬의 크기(모양) → `data.shape` : (2,3)
- dtype: 자료형/구조 → `data.dtype` : float64

```python
# 랜덤 데이터 생성
import numpy as np

data = np.random.randn(2,3)
print(data)
print(data*10)
print(data+data)
```

### 1, 2차원 ndarray 배열 생성

- ndim: 배열의 차원 → `data.ndim` : (2)

```python
import numpy as np

# 1차원 배열
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1) # ndarray 배열 생성
print(arr1)
print(arr1.ndim) # 차원
print(arr1.shape) # 행렬 개수 (행, 열)
print(arr1.dtype) # 지료형 타입

# 다차원 배열
data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
print(arr2)
print(arr2.ndim) # 배열의 차원 리턴 (2)
print(arr2.shape) # 행렬 개수 (2,4)
print(arr2.dtype) # 자료형 타입 (float64)

```
<br>

### zeros(), ones(), empty(), arange()로 초기화

```python
# 초기화
arr3 = np.zeros((2,3)) # 2x3 행렬의 값을 0으로 초기화
print(arr3)
print(arr3.ndim)

arr3 = np.ones((2,3)) # 1로 초기화
print(arr3)
print(arr3.ndim)

# 초기화 없이 배열 생성
arr4 = np.empty((2,3))
print(arr4)

# 리스트 대신 배열을 사용해 값 리턴
# 정렬된 배열 요소로 리턴
arr = np.arange(5)
print(arr)

```
<br>

### 산술 연산

- 모든 데이터를 행렬 연산으로 처리
- 각 요소에 대한 산술 연산과도 같다.

```python
arr = np.array([[1,2,3],[4,5,6]])
print(arr)
# 대응되는 각 요소 곱하여 제곱의 값 생성
arr2 = arr*arr 
print(arr2) 

arr2 = 1/arr
print(arr2)
arr2 = np.array([[0,4,1],[7,2,12]])
val = arr2>arr # 각 요소별로 비교 연산 수행
print(val)
```
<br>

### 인덱싱과 슬라이싱

```python
arr = np.arange(10)
print(arr)

print(arr[5]) # indexing
print(arr[5:8]) # slicing

arr[5:8] = 12 # 각 요소를 모두 12로 변경
arr[:] = 10 # 모든 요소를 10으로 변경
```
<br>

### 다차원 배열의 인덱싱

```python
import numpy as np

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[2], arr2d[0][2])
print(arr2d[0,2])

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d)
print(arr3d[0])
```

```python
old = arr3d[0].copy() # 첫번째 배열을 복사
print(old)
print(arr3d[1, 0]) # 두번째 배열의 0번째 값
print(arr3d[1,0,2]) # 두번쨰 배열의 0행 2열 값
```

<br>

### 불린 인덱싱 (Boolean Indexing)

```python
names = np.array(['Bob', 'Joe', '‘Will', 'Bob','Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4) #generate normal distribution

print(names=='Bob') # 각 요소와 비교
print(data[names=='Bob']) # boolean이 True인것만 

mask = (names=='Bob') | (names=='Will') # Bob 또는 Will

print(mask)
print(data[mask])
```
<br>

### 팬시 인덱싱(Fancy Indexing) **

- 정수형 배열을 사용하여 인덱스를 나타냄
- `배열을 인덱스로 넣어 원하는 값을 추출하기`

```python
array = np.empty((8,4))
for i in range(8):
  array[i]=i # 각 행의 요소에 i값을 모두 할당
print(array)

# 1차원 배열을 인덱스로 (행 추출)
print(array[[4,3,0,6]]) # 특정한 값만 추출 (인덱스 범위를 4,3,0,6 행에 해당하는 행만 추출)

# 0부터 31까지 값을 할당하고, 8x4 모양으로 변경 
array = np.arange(32).reshape((8,4))
print(array)

# 2차원 배열을 인덱스로 넣어 원하는 값 추출 (행,열값 추출)
# (1,0), (5,3), (7,1), (2,2)
print(array[[1,5,7,2],[0,3,1,2]])
```
<br>

### Transpose - 행렬의 전치 **

- `이해 필요 !!`
- 행렬끼리의 곱셈이 필요한 경우
    - 첫번째 행렬의 행과 두번째 행렬의 열, 첫번째 행렬의 열과 두번째 행렬의 행의 값이 동일해야 함

```python
arr = np.arange(15).reshape((3,5))
print(arr)

print(arr.T) # transpose (행과 열을 변경)

arr = np.random.randn(6,3)
print(arr)
val = np.dot(arr.T, arr) # 전치행렬의 연산
print(val)
```

### 유니버셜 함수

- 행렬 연산을 위한 내장 함수
- 빠른 벡터화된 래퍼 제공
- `np.sqrt(배열)`
- `np.exp(배열)`

```python
array = np.arange(10)
print(array)
print(np.sqrt(array)) # 각 요소를 제곱
print(np.exp(array)) # e^() 에 들어가야 할 값을 array로 넘겨줌
```

- `np.maximum(배열1, 배열2)`

```python
x = np.random.randn(8)
y = np.random.randn(8)
print(x,y)

val = np.maximum(x,y) # 비교하여 더 큰 값을 추출
print(val)
```

- `np.modf(배열)` 로 나머지와 몫 구하기