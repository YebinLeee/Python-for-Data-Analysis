# 6. Pandas 라이브러리

# Pandas 자료구조

- 파이썬에서 쉽고 빠르게 데이터를 분석하고 다루기 위한 자료구조와 도구들을 포함함
- 기존 python 라이브러리와 가장 큰 차이는, 테이블 기반 및 이종의 데이터를 다루기 위한 함수를 지원한다는 점

```python
import pandas as pd
from pandas import Series, DataFrame
```

<br>

## Series 자료구조

- 원소 값의 순서를 포함하는 배열과 같은 1차원 객체를 나타냄

```python
def SeriesFunction():
    obj = pd.Series([4,7,-5,3]) # 배열을 키(인덱스)-값 쌍의 사전 데이터로 변경
    print(obj) # key-value 모두 출력
    print(obj.values) # value들 출력
    print(obj.index) # 인덱스 key들 출력
    
    # 인덱스 키값을 직접 부여
    obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])
    print(obj2)
    print(obj2.index)
    
    # 단일 원소값 선택 또는 원소 집합 선택을 위해 인덱스의 레이블 사용
    print(obj2['a']) # 'a' 인덱스에 해당하는 -5 출력
    obj2['d']=6 # 'd' 인덱스에 해당하는 4를 6으로 변경
    print(obj2)
    
```


### Numpy의 기능 사용

```python
def SeriesNumpy():
    # Series자료구조를 가진 객체에 Numpy 함수 기능 적용
    obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])
    print(obj2[obj2>0]) # value가 0보다 큰 경우 출력
    print(obj2*2)
    print(np.exp(obj2))
```



### 사전 자료형

```python
# 사전 자료형으로 구성된 많은 데이터에 대해 적용하여 사용
def SeriesDictionary():
    sdata = {"Ohio":35000, "Texas":71000, "Oregon":16000, "Utah":5000}
    obj3 = pd.Series(sdata) # 사전 자료형인 sdata로부터 Series 객체 생성
    print(obj3)
    
    # 사전 자료형으로 시리즈 객체 생성 시 사전의 키 값이 순서대로 시리즈 객체의 인덱스로 지정
    # 인덱스를 찾을 수 없는 경우 NaN(not a number)로 표시
    states = ["California", "Ohio", "Oregon", "Texas"]
    obj4 = pd.Series(sdata, index=states)
    print(obj4)
    
    # missing data 탐색
    print(pd.isnull(obj4))
    print(pd.notnull(obj4))
```

---


## DataFrame 자료구조

- 여러 Column으로 구성된 테이블 방식의 데이터를 표현
- 각 컬럼은 서로 다른 형태의 데이터를 표현할 수 있고, row와 column에 대한 인덱스를 가짐
- `columns` , `index` 속성

```python
def DataFrameFunc():
    # 각 열에 대한 데이터들
    data = {
        "state" : ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year" : [2000,2001,2002,2001,2002,2003],
        "pop" : [1.5,1.7,3.6,2.4,2.9,3.2]
    }
    frame = pd.DataFrame(data) # data를 판다스의 데이터프레임 객체로 생성
    print(frame)
    print(frame.head()) # 첫 5개 행의 정보를 추출
    
    # columns를 지정하면 데이터를 지정된 순서로 관리 가능
    frame2 = pd.DataFrame(data, columns=["year","state","pop"])
    print(frame2)
    
    # 사전과 같은 표기법 또는 속성 값으로 특정 데이터에 접근 가능
    print(frame2["state"])
    
    # 로우는 loc 속성을 가지고 이름이나 위치로 데이터에 접근 가능
    frame3 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"],
                          index=["one", "two", "three", "four", "five","six"])
    print(frame3)
    
    
    # 비어있는 칼럼에는 스칼라 값이나, 배열의 값이 할당 될 수 있음
    frame3["debt"] = np.arange(6)
    print(frame3)
    
    # 배열 값 할당
    val = pd.Series([-1.2, -1.5, -1.7], index=["two", "four", "five"])
    frame3["debt"]=val 
    print(frame3)
    
    # 칼럼 추가
    frame3["eastern"]=(frame3.state=="Ohio") # eastern 칼럼 추가 (Ohio state인 경우에만 True값 부여)
    print(frame3)
    # 칼럼 삭제
    del frame3["debt"]
    print(frame3)
```

### 인덱스 객체(Index Objects)

- Pandas 인덱스 객체는 테이블 형식의 데이터에서, 각 축에 대한 레이블과 메타 데이터를 다룸
- Series, Data Frame 객체 생성할 때 배열이네 레이블의 순서가 내부적으로 인덱스로 변환됨
- `reindex()` 함수를 통해 인덱스를 새로 추가(NaN값으로 fill)하거나 순서를 재배치 가능, 또는 method의 ffill  값으로 데이터가 없는 부분을 보간하거나 채울 수 있음

```python
def PandasIndexes():
    # 인덱스 객체
    obj = pd.Series(range(3), index=['a', 'b', 'c'])
    print(obj, obj.index)
    
    # 재색인(Reindex)
    obj2 = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
    print(obj2)
    # 인덱스의 배열을 변경 (없던 인덱스 추가 시 NaN 추가)
    obj3 = obj2.reindex(['a', 'b', 'c', 'd', 'e'])
    print(obj3)
    
    # 시계열 데이터처럼 순서를 가진 데이터에 대해, 재색인은 값을 보간하거나 채워넣을 수 있음
    # 값이 없는 데이터에 대해, 채워넣을 수 있음
    # ffil 메소드를 사용해 이전에 채워진 값으로 채우는 것이 가능
    obj = pd.Series(['blue', 'purple', 'yellow'], index=[0,2,4])
    obj4 = obj.reindex(range(6), method='ffill')
    print(obj, '\n', obj4)
    
    # 재색인은 (로우) 인덱스 및 컬럼 변경 가능
    # 인덱스 순서에 대한 값을 전달하면, 로우에 대한 인덱스 재색인 (재배치)
    frame = pd.DataFrame(np.arange(9).reshape(3,3),
                         index=['a','b','c'],
                         columns=['Ohio', 'Texas', 'California'])
    print(frame)
    frame2 = frame.reindex(['a', 'b', 'c', 'd']) # d 칼럼 추가
    print(frame2)
    
    # columns 키워드를 사용해 DataFrame 컬럼 재색인 가능
    states = ['Texas','Utah','California'] # Ohio 컬럼은 제외, Utah를 새로 생성
    frame2 = frame.reindex(columns=states)
    print(frame2)
```

### 엔트리 삭제 (Dropping entries from an ax

- `drop` 메소드를 통해 축으로부터 지정된 값(칼럼)을 삭제한 새로운 객체 리턴

```python
def DroppingEntires():
    obj = pd.Series(np.arange(5.), index=['a','b','c','d','e'])
    print(obj)
    
    new = obj.drop('c') # C 컬럼 삭제      
    print(new)
    new = obj.drop(['d','c']) # 동시에 여러 컬럼 삭제
    print(new)
    
    data = pd.DataFrame(np.arange(16).reshape((4,4)),
                        index=['Ohio','Colorado','Utah','New York'],
                        columns=['one','two','three','four'])
    print(data)
    d1 = data.drop(['Colorado', 'Ohio'])
    print(d1)
```

<br><hr>

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
