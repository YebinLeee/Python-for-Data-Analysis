# 7. Pandas 2

# Pandas 필수 기능

## 인덱싱, 선택, 필터링(Indexing, Selection, Filtering)

- Series 인덱싱은 NumPy 배열 인덱싱과 유사하게 동작
- 인덱싱을 위해 정수값 대신 Series의 인덱스 사용 가능

```python
import numpy as np
import pandas as pd

# Pandas의 Series데이터의 인덱싱, 셀렉션, 필터링
def indexing():
    obj = pd.Series(np.arange(4.), index=['a','b','c','d'])
    print(obj)
    print(obj['b']) # 직접 지정한 인덱스로 접근
    print(obj[2:4]) # 정수 인덱스로 이용한 접근
    print( obj[[1,3]])
    print(obj['b':'d']) # 인덱스로 슬라이싱
```

- 행에 대한 인덱스 `index` , 열에 대한 인덱스  `columns`
- `loc` 과 `iloc` 으로 행, 열에 매칭되는 데이터 값 접근하여 필터링하기

```python
def indexing2():
    data = pd.DataFrame(np.arange(16).reshape((4,4)),
                        index=['Ohio', 'Colorado', 'Utah', 'New York'], # 행에 대한 인덱스 명시
                        columns=['one','two','three','four']) # 열에 대한 인덱스 명시
    print(data)
    print(data['two']) # two 에 해당하는 열과 인덱스 출력
    print(data[:2])
    
    # 축 레이블(loc) 또는 정수(iloc) 을 이용해 객체의 로우와 컬럼 하위 집합 선택
    d = data.loc['Colorado', ['two','three']] # Colorado 행 데이터 중 two, three 열의 값 저장
    print(d)
    d = data.iloc[2,[3,0,1]] # 2행 데이터 중 3,0,1 열의 값 저장
    print(d)
```

<br>


## 산술연산과 정렬 (Arithmetic & Data Alignment)

- 서로 다른 인덱스를 가진 객체들 사이에서 상술연산 수행이 가능
- value가 있어야 하며, 한쪽이라도 없는 경우 NaN 결과값이 도출됨

### 산술연산 메소드로 값 채우기

- 특정 값으로 채우기
    - 로우 인덱스와 컬럼 이름을 사용해 지정된 위치의 값 변경
    - 산술연산 메소드를 이용해 객체의 데이터 연산
    - `df.loc[1,'b'] = np.nan` → 1번 행, ‘b’번 열에 수정할 값을 할당

```python
def arithemtic():
    s1 = pd.Series([7.3,-2.5,3.4,1.5], index=['a','c','d','e'])
    s2 = pd.Series([-2.1,3.6,-1.5,4,3.1], index=['a','c','e','f','g'])
    print(s1+s2)
    
    df1 = pd.DataFrame(np.arange(9.).reshape((3,3)),
                       columns=list('bcd'),
                       index=['Ohio','Texas','Colorado'])
    df2 = pd.DataFrame(np.arange(12.).reshape((4,3)),
                       columns=list('bde'),
                       index=['Utah','Ohio','Texas','Oregon'])
    print(df1+df2)
    
    # 특정 값으로 채우기
    df1 = pd.DataFrame(np.arange(12.).reshape((3,4)),
                      columns=list('abcd'))
    df2 = pd.DataFrame(np.arange(20.).reshape((4,5)),
                       columns=list('abcde'))
        
    print(df1)
    print(df2)
    
    df2.loc[1,'b'] = np.nan # 1번 행, 'b'번 열의 데이터에 nan 값 넣기
    
    df = df1.add(df2, fill_value=0) # 값이 없는 부분은 0으로 채워서 더하기
    print(df)

    print(1/df) # 나누어질 수 없는 경우는 inf
    print(df.rdiv(1)) # reverse한 후 분자를 1로 설정
```

<br>


## DataFrame과  Series 사이 연산

- DataFrame과 Series사이의 산술 연산은 DataFrame의 컬럼에 있는 Series의 인덱스와 일치 시키고, 그 로우에 대한 값을 브로드캐스팅
    - `브로드캐스팅`  : 전체 영역에 연산을 적용

```python
def arithmetic2():
    frame = pd.DataFrame(np.arange(12.).reshape((4,3)),
                         columns=list('bde'),
                        index=['Utah','Ohio','Texas','Oregon'])
    series = frame.iloc[0] # 0번째 행
    print(frame)
    print(series)
    
    # 브로드캐스팅 : 전체 영역에 연산을 적용
    
    # 0번째 행의 데이터 값들을 모든 데이터 값에 대하여 subtract
    val = frame - series 
    print(val)
    
    series2 = frame['d'] # d번 열
    print(series2)
    
    # 'd'번 열의 값들을 모든 열의 데이터 값에 대하여 subtract 
    val = frame.sub(series2, axis='index')
    print(val)
```

### 함수 응용과 매핑 (Func. Application & Mapping)

- NumPy의 유니버셜 함수(원소 단위의 배열 처리 함수)와 Pandas 객체를 함께 동작
- 1차원 배열에 대한 함수를 각 컬럼과 로우에 적용하는 동작도 자주 활용됨 (람다 함수 매핑)
- `apply()` 메소드에 `axis=columns` 를 사용하면 주어진 함수를 각 로우마다 한번씩 적용 가능 (행)

```python
def applicationMapping():
    frame = pd.DataFrame(np.random.randn(4,3),
                         columns=list('bde'),
                         index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
    print(frame)
    print(np.abs(frame)) # 모든 값의 절대값 구하기
    
    # 람다 함수 사용
    f = lambda x : x.max() - x.min() # 각 칼럼별 데이터들 중 최대값과 최소값에 대한 차이
    v = frame.apply(f)
    print(v)
    
    v  = frame.apply(f, axis='columns') # 각 행의 데이터들 중 최대값 최소값에 대한 차이
    print(v)
    
    # 원소 단위의 python 함수를 pandas와 함게 사용ㄴ
    def f(x):
        return pd.Series([x.min(), x.max()], index=['min','max'])
    v = frame.apply(f)
    print(v)
```

### 정렬과 랭킹 (Sorting & Ranking)

- 정렬

```python
def sorting():
    obj = pd.Series(range(4), index=['d','b','a','c'])
    print(obj)
    print(obj.sort_index()) # 인덱스를 기준으로 sort
    
    frame = pd.DataFrame(np.arange(8).reshape((2,4)),
                         index=['three','one',],
                         columns=['d','a','b','c'])
    print(frame)
    print(frame.sort_index())
    print(frame.sort_index(axis=1)) # 칼럼에 대한 정렬
```

- ranking

```python
# 1부터 오름차순으로 값을 부여
def ranking():
    obj = pd.Series([7,-5,7,4,2,0,4]) # 같은 값이 여러 개 있는 경우 평균값으로 랭크 값을 부여 (3,4번째인 경우 3.5, 3.5 부여)
    print(obj.rank()) # default 옵션 (average)
    
    # 동일한 값이 있을 때 먼저 마주치는 데이터에 랭크를 먼저 부여
    print(obj.rank(method='first'))
```

<br>


### 중복 인덱스

- 모든 Pandas함수들이 독립적인 레이블을 가지도록 하지만, 경우에 따라서는 중첩된 레이블(인덱스)를 가질 수 있음

```python
def sameLabel():
    obj = pd.Series(range(5), index=['a','a','b','b','c'])
    print(obj)
    print(obj.index.is_unique) # 모든 레이블이 unique한가? -> False
    print(obj['a'])
```

<br><hr><br>

# 통계

- Pandas 객체는 수학 및 통계 메소드를 포함하고 있음
- 대부분 통계 메소드는 Series나 DataFrame에 대한 축소와 요약의 통계 범주에 속함


```python
def pandasStat():
    df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                       [np.nan, np.nan],[0.75, -1.3]],
                      index=['a','b','c','d'],
                      columns=['one','two'])
    print(df)
    print(df.sum()) # 열 단위 합계
    print(df.sum(axis='columns')) # 행 단위 합계
    print(df.mean(axis='columns',skipna=False)) # 행 단위 평균값 (NaN 인 경우 연산 불가)
    print(df.idxmax()) # 열 단위 최댓값의 인덱스
    print(df.cumsum()) # 열 단위 누적합 계산
    print(df.describe()) # 행칼럼에 대한 설명
```
