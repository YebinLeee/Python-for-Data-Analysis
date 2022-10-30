# 4. Numpy

# NumPy 배열이란?

- `Numpy` : Numerical Python, 파이썬에서 산술 계산을 위한 가장 중요한 필수 패키지 중 하나

### 특징

- `ndarray` : 배열 중심의 빠른 산술 연산을 지원하는 다차원 배열, 브로드캐스팅 기능 지원 (행렬 연산)
- 반복되는 루프를 가지지 않고, 전체 배열에서의 빠른 연산을 지원
- 디스크에 배열 데이터를 읽고 쓰기 위한 도구와 메모리에 적재된 파일을 다루는 도구 지원
- 선형 대수, random number 생성, 푸리에 변환 등 고급 수학 연산 지원
- C, C++ 로 작성된 라이브러리를 C API를 사용해 NumPy 와 연동

<br>


### 데이터 분석을 위한 NumPy

- 벡터 배열 상에서 데이터 가공, 정제, 부분집합, 필터링, 변형과 빠른 배열 연산 제공
- 정렬, 유일 원소 찾기, 집합 연산같은 일반적인 배열  처리알고리즘 지원
    - sort, unique, set operation
- 효율적인 데이터 통계적 표현 및 다양한 종류의 데이터 병합 및 데이터 간의 관계 조작, 요약 기능 제공
- 반복 루프를 사용하지 않고 배열의 조건절 표현을 허용하는 배열 처리 기능
- 그룹(데이터 묶음) 단위의 수집, 변형, 함수 적용 등의 데이터 처리 지원
    - aggregation, transformation, function application

<br>


# 1. ndarray : 다차원 배열 객체

- 모든 배열 요소는 동일한 자료형을 가짐
- 전체 데이터 블록에 수학적 연산을 수행할 수 있게 해줌
- **********************메타데이터**********************를 담는 특수한 객체들 : dtype
- `ndarray.shape` : 행렬의 크기(모양) (행,열)
- `ndarray.dtype` : 자료형을 알려주는 객체
    - 명시적으로 지정하지 않는 한 자동적으로 적절한 자료형을 추론함
    - 기본적으로 float64(부동소수점)으로 판단

```python
# 랜덤 데이터 생성
import numpy as np

data = np.random.randn(2,3)

print(data)
print(data*10)
print(data+data)

print(data.shape) # (2,3)
print(data.dtype) # float64
```

### 1, 2차원 ndarray 배열 생성

- `np.array()` 메서드 사용해여 순차적인 개게를 넘겨줌
- `ndarray.ndim` : 배열의 차원 (1차원, 2차원)
- 같은 길이를 가지는 다차원 배열 또한 변환 가능

```python
import numpy as np

# 1차원 배열
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1) # ndarray 배열 생성
print(arr1)

# 다차원 배열
data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
print(arr2)

print(arr2.ndim) # 배열의 차원 리턴 (2)
print(arr2.shape) # 행렬 개수 (2,4)
```

### zeros(), ones(), empty(), arange()로 초기화

- `arange()` : 파이썬의 range 함수의 배열 버전
- `empty()` : 가비지 값으로 채워진 배열 반환

```python
print(np.zeros(10)) # [0. 0. 0. ... 0.]
# 초기화
arr3 = np.zeros((2,3)) # 2x3 행렬의 값을 0으로 초기화
print(arr3)

arr4 = np.ones((2,3)) # 1로 초기화
print(arr4)

# 초기화 없이 배열 생성
arr5 = np.empty((2,3))

# 리스트 대신 배열을 사용해 값 리턴
# 정렬된 배열 요소로 리턴
arr6 = np.arange(15) # [0. 1. 2. ... 14]

arr7 = np.array([1,2,3], dtype=np.float64)
```

### 캐스팅, 형변환

- ndarray의 `astype()` 메서드를 사용해 배열의 dtype을 다른 형으로 명시적으로 변환(캐스팅) 가능
    
    ```jsx
    arr = np.array([1,2,3,4,5])
    print(arr.dtype) # 'int64'
    
    float_arr = arr.astype(np.float64) # float64 형으로 변경
    print(float_arr.dtype) # 'float64'
    ```
    
- 부동소수점을 정수형 dtype으로 변환 시 소수점 아래 자리는 버려짐 (3.7 → 3, -2.6 → -2)
- 숫자 형식의 문자열은 숫자로 변환 가능
    
    ```jsx
    numeric_strings = np.array(['1.5', '-2.6', '-0.0003'])
    print(numeric_strings.astype(float))
    ```
    

### 산술 연산

- **************벡터화**************: for문을 작성하지 않고 데이터를 일괄 처리할 수 있는 특징
- 같은 크기의 배열 간의 산술 연산은 배열의 각 원소 단위로 작용

```python
arr = np.array([[1,2,3],[4,5,6]])
print(arr)

arr2 = arr*arr # 대응되는 각 요소 곱하여 제곱의 값 생성
print(arr2) 

arr2 = 1/arr
arr2 = np.array([[0,4,1],[7,2,12]])
val = arr2>arr # 각 요소별로 비교 연산 수행
print(val)
```

### 색인 : 인덱싱과 슬라이싱

- Numpy 배열의 부분집합이나 개별 요소를 선택하기 위해 사용
- ********************************브로드캐스팅:******************************** `arr[5:8]=12` 처럼 배열 조각에 스칼라값을 대입하면 12가 선택 영역 전체로 전파되는 것
    - 일반적인 리스트와 다르며, 데이터가 복사되지 않고 뷰에 대한 변경은 그대로 원본 배열에 반영된다.
    - 대용량 데이터 처리를 염두하여 설계되었기 때문에 데이터 복사가 아닌 실제 원본 데이터 변경임을 잊지 말자.
    - 뷰 대신 슬라이싱한 복사본을 얻고 싶다면 `arr[5:8].copy()` 를 사용하여 명시적으로 배열을 복사해야 함

```python
arr = np.arange(10)
print(arr)

print(arr[5]) # indexing
print(arr[5:8]) # slicing

arr[5:8] = 12 # 해당 위치의 원소 값들을 12로 변경
arr[:] = 10 # 모든 요소를 10으로 변경
```

### 다차원 배열의 인덱싱

- 2차원 배열에서의 색인에 해당하는 요소는 1차원 배열이다.
- 개별 요소에 접근하기 위해서 [row, column] 방식을 사용할 수 있다.
- 2차원 numpy 배열에서 `arr[0][2]` 는 `arr[0,2]`로 표현 가능하다.

```python
import numpy as np

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[2], arr2d[0][2], arr2d[0,2]) # 3번째 행, 첫번째 행의 3번째 열 ([7,8,9], 3)

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d[0]) # 2차원 배열
```

```python
old = arr3d[0].copy() # 첫번째 배열을 복사
print(arr3d[1, 0]) # 두번째 배열의 0번째 값
```

- 인덱싱과 슬라이싱을 함께 사용하는 경우 다음과 같이 표현 가능하다.

<br>

### 불린 인덱싱 (Boolean Indexing)

- 중복된 이름이 포함된 배열에서 어떠한 값과 같은지 확인하기 위해 비교연산자를 통해 모든 배열의 원소와 비교를 가능하다.
- 아래 예시에서 각각의 이름은 data 배열의 각 로우에 대응한다. True를 갖는 행값의 데이터가 추출된다.
    
    ```python
    names = np.array(['Bob', 'Joe', '‘Will', 'Bob','Will', 'Joe', 'Joe'])
    data = np.random.randn(7, 4) #generate normal distribution
    
    print(names=='Bob') # 각 요소와 비교 [True False False True ...]
    print(data[names=='Bob']) # boolean이 True인것만 (0번, 3번 로우 출력)
    print(data[name=='Bob', 2:] # 각 로우에서 2번 칼럼부터 추출
    print(data[~names=='Bob']) # ~연산자를 이용해 조건절 부인
    
    cond = names=='Bob'
    print(data[~cond])
    ```
    
- 세 가지 이름 중 두 가지 이름을 선택하려면 `&` 연산자나 `|` 연산자 사용 가능
    
    ```python
    mask = (names=='Bob') | (names=='Will') # Bob 또는 Will
    
    print(mask)
    print(data[mask])
    ```
    
- 조건을 각 배열 요소에 적용하기
    
    ```python
    data[data<0] = 0 # 모든 음수를 0으로 바꾸기
    data[names!='Joe'] = 8 'Joe` 를 제외하고 모두 8로 변경하기
    ```
    

### 팬시 인덱싱(Fancy Indexing) **

- ************************팬시 색인(fancy indecing)************************ :
- 정수형 배열을 사용하여 인덱스를 나타냄
- `배열을 인덱스로 넣어 원하는 값을 추출하기`

```python
array = np.empty((8,4))
for i in range(8):
  array[i]=i # 각 행의 요소에 i값을 모두 할당
print(array)

# 1차원 배열을 인덱스로 (행 추출)
print(array[[4,3,0,6]]) # 특정한 값만 추출 (인덱스 범위를 4,3,0,6 행에 해당하는 행만 추출)

# 음수 색인
print(arr[[-3, -5, -7]]) # 5, 3, 1에 해당하는 로우 출력

# 0부터 31까지 값을 할당하고, 8x4 모양으로 변경 
array = np.arange(32).reshape((8,4))
print(array)

# 2차원 배열을 인덱스로 넣어 원하는 값 추출 (행,열값 추출)
# (1,0), (5,3), (7,1), (2,2)
print(array[[1,5,7,2],[0,3,1,2]])

# 1,5,7,2번 행을 모드 출력함과 동시에, 순서를 0,3,1,2번으로 정하고자 할 때
print(array[[1,5,7,2]][:,[0,3,1,2]]) 
```

### 배열 전치와 축 바꾸기 (Transpose) **

- `transpose()` 메서드와 T 속성을 통해 데이터의 모양이 바뀐 뷰를 반환
- 행렬의 내적 : `np.dot(arr1, arr2)` : 행렬끼리의 곱셈이 필요한 경우
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

<br>

# 2. 유니버셜 함수

- 행렬 연산을 고속으로 수행하는 벡터화된 래퍼 함수
- 빠른 벡터화된 래퍼 제공
- ********************************************단항 유니버셜 함수********************************************
    - `np.sqrt(배열)` : 제곱근 구하기
    - `np.exp(배열)`
        
        ```python
        array = np.arange(10)
        print(array)
        print(np.sqrt(array)) # 각 요소를 제곱
        print(np.exp(array)) # 각 원소의 e^x 를 구함
        ```
        
- ********************************************이항 유니버셜 함수********************************************
    - `np.maximum(배열1, 배열2, ...)` : 각 원소를 비교하여 가장 큰 값을 계산
        
        ```python
        x = np.random.randn(8)
        y = np.random.randn(8)
        print(x,y)
        
        val = np.maximum(x,y) # 비교하여 더 큰 값을 추출
        print(val)
        ```
        
    - `np.modf(배열)` 로 나머지와 몫 구하기
        - `divmod` 의 벡터화 버전, 분수를 받아 몫과 나머지를 함께 반환함


<br>


# 3. 배열을 이용한 배열 지향 프로그래밍 (Array oriented programming)

- `sqrt(x^2 + y^2)` :
- `np.meshgrid(point1, point2)` : 두 개의 1차원 배열을 받아 가능한 모든 (x,y) 짝을 만 수 있는 2차원 배열 두개를 반환함

```python
import matplotlib.pyplot as plt

def array_oriented_sqrt():
    points = np.arange(-5, 5, 0.01) # -5에서 5사이의 0.01 간격의 값들을 points 배열에 담음
    
    xs, ys = np.meshgrid(points, points) # 모든 (x,y) 쌍 생성
    print('xs:', xs, 'ys:', ys)

    z=np.sqrt(x**2+y**2) # 루트 제곱근 값을 z에 대입
    print(z)

    plt.imshow(z, cmap=plt.cm.gray) # z의 값들을 흑백 matplot 생성
    plt.colorbar()
    plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
    plt.show()
```

### 배열 연산에 대한 조건부 표현식

- `np.where()` : x if 조건 else y 와 같은 삼항식의 벡터화된 버전
    - where(condition, x, y)

```python
def case_expression():
    # case 1
    xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    cond = np.array([True, False, True, True, False])
    
		# cond값이 True인 경우에는 x 아니면 y
		result = [(x if c else y for x,y,c in zip(xarr, yarr, cond))] 
    
		# where 함수 사용하여 간단히 구현
		result = np.where(cond, xarr, yarr) 
    print(result)
    
    # case 2
    array = np.random.randn(4, 4)
    print(array)
    print(array > 0)
    
    # 값이 양수면 2, 아니라면 -2
    ret = np.where(array > 0, 2, -2)
    print(ret)
```

### 수학, 통계 메서드

- `np.mean()`  : 평균
- `np.sum()` : 합
- `np,.cumsum()` : 누적 합
- `np.cumpord()` : 누적 곱
- 속성값으로 `axis=0` 인 경우 로우(행) 단위, `axis=1` 인 경우 칼럼(열) 단위

```python
import numpy as np

arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(arr.mean()) # 평균값
print(np.mean(arr))

print(arr.sum()) # 전체 합
print(arr.sum(axis=1)) # 행에 대한 합 계산 (axis=1)
print(arr.sum(axis=0)) # 열에 대한 합 계산 (axis=0)

arr = np.array([0,1,2,3,4,5,6]) # 현재 요소까지의 누적합
print(arr.cumsum())

arr = np.array([[0,1,2], [3,4,5], [6,7,8]])
print(arr.cumsum(axis=0)) # 각 행(로우) 대한 누적 합
print(arr.cumprod(axis=1)) # 각 열(칼럼)에 대한 누적 곱
```

### 불린 배열을 위한 메서드

- 이전 메서드의 불리언 값을 1(True) 또는 0(False)로 강제하여, `sum()` 메서드 사용 시 배열에서 True 원소의 개수를 알아낼 수 있음
- `np.any()` : True가 하나라도 존재하는가 (0이 아닌 원소는 True라 간주)
- `np.all()` : 모든 원소가 True 인가

```python
# 불린 배열을 위한 메소드
def boolean_array_method():
    arr = np.random.randn(100)
    print(arr)
    print((arr>0).sum()) # 0보다 큰 숫자의 개수
    
    bools = np.array([False, False, True, False])
    print(bools.any()) # 배열 원소 중에 true기 있는지
    print(bools.all()) # 모든 배열 원소가 true인가
```

### 정렬

- `np.sort()` 메서드로 정렬

```python
# 정렬
def sort_method():
    arr = np.random.randn(6)
    print(arr)
    print(np.sort(arr)) # 오름 차순 정렬
    
    # 2차원배열 정렬
    arr = np.random.randn(5,3)
    arr.sort(1) # 1차원 부분을 정렬 (각 행별로 오름차순 정렬)
    print(arr)
```

### 집합 관련 함수

- `np.unique()` : 중복된 원소를 제거하고 남은 원소를 정렬된 형태로 반환
- 파이썬에서는 `sorted(set(arr))` 와 같이 구현
- `np.in1d(arr1, arr2)` : arr1의 원소가 arr2의 원소를 포함하는지 나타내는 불리언 배열 반환

```python
# 집합에서의 Unique()
def unique_method():
    names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joel"])
    print(np.unique(names)) # 중복되는 것은 1번만 (오름차순으로 정렬하여 출력)
    
    ints = np.array([3,3,3,2,2,1,1,4,4])
    print(np.unique(ints))
    
    values = np.array([6,0,0,3,2,5,6])
    print(np.in1d(values, [2,3,6])) # 인자로 주어진 배열의 원소가 포함되어있는지 확인 후 Bool로 리턴
```

<br>

# 4. 배열 데이터의 파일 입출력

- `np.save(arr_name, arr)` : arr_name으로 지정하여 배열 arr을 디스크에 저장
    - 압축되지 않은 원시 바이너리 형식의 .npy 파일로 저장됨
- `np.load('arr_name.npy')` : 배열을 불러옴
- `np.savez('arr_name', a=arr, b=arr)` : 여러 배열을 압축된 형식으로 저장
    - 저장하려는 배열을 키워드 인자 형태로 전달
    - 각각의 배열을 필요할 때 불러올 수 있도록 사전 형식의 객체에 저장함

```python
# 배열 데이터의 파일 입출력
def fileIO_array():
    arr = np.arange(10)
    np.save("some_array", arr) # 바이너리 포맷 파일로 데이터 저장
    ld = np.load("some_array.npy") # load, save 메소드로 파일 입출력 처리 가능
    print(ld)
    
    # savez로 압축되지 않은 묶음 형식으로 다중 배열을 저장
    np.savez("array_archive.npz", a=arr, b=arr) # a와 b에 배열 저장
    arch = np.load("array_archive.npz") # 배열 데이터는 사전 형식의 객체로 불러옴
    print(arch['b']) # b에 저장한 배열 출력
```

<br>

# 5. 선형 대수 (Linear Algebra)

- 배열 기반 프로그래밍에서 선형 대수는 매우 중요 (행렬 곱셈, 분할, 행렬식, 정사각 행렬 수학 등의 선형대수
- 행렬의 곱셈은 `*` 가 아닌 `np.dot()` 함수를 사용하자
- 파이썬 v3.5 이후부터 행렬곱 연산자로 @ 사용 가능 `print(x @ np.ones(3))`

```python
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[6,23],[-1,7],[8,9]])

ret = x.dot(y) # x와 y의 행렬 곱  => ret = np.dot(x,y)와 동일
print(ret)

print(x @ np.ones(3)))
```

- `numpy.linalg` 모듈은 행렬의 분할, 역행렬, 행렬식 등을 포함함

```python
import numpy as np
from numpy.linalg import inv

x = np.random.randn(3,3)
mat = x.T.dot(x) # x.t.의 전치 행렬과 x의 곱을 계산
print(mat)

# 정사각 행렬의 역행렬 구하기
print(inv(mat))
print(mat.dot(inv(mat)))
```

<br>

# 6. 난수 생성 (Random Number Generation)

- `numpy.random` 모듈은 파이썬 내장 random 함수를 보강하여, 다양한 종류의 확률분포로부터 효과적으로 표본값을 생성하는데 주로 사용됨
- 한 번에 하나의 값만 생성하는 파이썬 random 모듈과 다르게 `numpy.random`은 매우 큰 크기의 표본을 생성하는데 빠르다
- `norma()` 을 사용해 표준정규분포로부터 4x4 크기의 표본을 생성해보자.

```python
samples = np.random.normal(size=(4,4))
print(samples)
```

- **********************유사 난수 :********************** 난수 생성기의 시드값에 따라 정해진 난수를 알고리즘으로 생성한다.
    - `np.random.seed()` 를 사용해 난수 생성을 변경할 수 있다. 전역 난수 시드(global random seed) 를 사용
- 전역 상태를 피하기 위해 `numpy.random.RandomState` 를 사용해 다른 난수 생성기와 분리 가능하다.
    
    ```python
    np.random.seed(1234)
    
    rng = np.random.Randomstate(1234)
    ret = rng.randn(10)
    print(ret)
    ```
    

### Random Points (계단 오르내리기 예제)

- 배열 연산의 활용을 보여주는 간단한 애플리케이션을 구현해보자.
- 계단 중간에서 같은 확률로 한 계단 올라가거나 내려간다.
- 순수 파이썬으로 내장 random 모듈을 사용한 경우
    
    ```python
    import numpy as np
    import random
    
    position = 0
    walk = [position]
    steps = 1000
    for i in range(steps):
        step = 1 if random.randint(0,1) else -1
        position += step
        walk.append(position)
    
    plt.plot(walk[:100])
    plt.show()
    ```
    
- `np.random` 모듈을 사용해 1000번 +1. -1 수행한 결과를 한번에 저장하고 누적합 계산하기