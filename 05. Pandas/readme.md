# 5. Pandas

# Pandas 자료구조

- 파이썬에서 쉽고 빠르게 데이터를 분석하고 다루기 위한 자료구조와 도구들을 포함함
- 기존 python 라이브러리와 가장 큰 차이는, for문을 사용하지 않고 데이터를 처리하거나 배열 기반의 함수를 제공하는 등 NumPy 배열 기반 계산 스타일을 많이 차용하여, 테이블 기반 및 이종의 데이터를 쉽게 다룰 수 있게 된다는 점
    - NumPy와 다른 점은, pandas는 표 형식의 데이터나 다양한 형태의 데이터를 다루는데 초점을 맞춰 설계했다는 점 (Numpy는 단일 산술 배열 데이터를 다루는데 특화되어 있음)
- 다른 산술 계산 도구인 `Numpy` 와 `SciPy` , 분석 라이브러리인 `statsmodels` 과 `scikit-learn` , 시각화 도구인 `matplotlib` 과 함꼐 사용한다.

```python
import pandas as pd
from pandas import Series, DataFrame
```

<br>

# 1. pandas 자료구조 소개

## Series 자료구조

- `Series` : 일련의 객체를 담을 수 있는 1차원 배열 같은 자료구조이다.
- **********색인(index)********** 기능을 제공 (원소 값의 순서) - 왼쪽에는 색인을, 오른쪽에는 해당 색인의 값을 보여줌 (0~N-1)
- `obj.index` , `obj.values` 를 통해 인덱스와 값을 얻을 수 있다.
- `index` 속성을 통해 직접 색인을 지정 가능하다.
- 단일 값을 선택할 때에는 `obj['b']` , 여러 값을 선택할 때에는 `obj[['c', 'a', 'b']]` 와 같이 색인으로 라벨을 사용할 수 있다.

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

- 산술 곱셈 또는 수학 함수를 적용하는 등 Numpy 배열 연산을 수행해도 색인-값 연결이 유지된다.

```python
def SeriesNumpy():
    # Series자료구조를 가진 객체에 Numpy 함수 기능 적용
    obj2 = pd.Series([4,7,-5,3], index=['d','b','a','c'])
    print(obj2[obj2>0]) # value가 0보다 큰 경우 출력
    print(obj2*2)
    print(np.exp(obj2))
```



### 사전 자료형

- Series는 고정 길이의 정렬된 사전형이라고 생각할 수 있다. 색인 값에 데이터 값을 매핑하고 있으므로 파이썬의 사전형과 비슷하다.
- 파이썬 사전 객체로부터 Series 객체 생성 가능
- 직접 배열을 index 속성으로 넣어 색인을 직접 지정할 수 있다. (해당 키에 해당하는 값이 순서대로 나타남)
- `pd.isnull(obj)` , `pd.notnull(obj)`: 누락된 데이터가 있는지 불리언 배열로 확인 가능

```python
# 사전 자료형으로 구성된 많은 데이터에 대해 적용하여 사용
def SeriesDictionary():
    sdata = {"Ohio":35000, "Texas":71000, "Oregon":16000, "Utah":5000}
		print("Ohio" in sdata) # True
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

		# 산술 연산 시 자동으로 키들을 매핑하여 연산 가능
		print(obj3 + obj4)
```


- `name` 속성을 지정 가능
- Series의 색인은 직접 대입하여 변경도 가능하다.

```python
obj4.name = 'population' # Series의 객체 이름 지정
obj4.index.name = 'state' # 인덱스 이름 지정

obj.index = ['Bob', 'Joe', 'Seve', 'Ryan']
```

<br>

## DataFrame 자료구조

- 표 같은 스프레드 시트 형식의 자료구조이며, 여러 개의 컬럼이 있는데 각 칼럼은 서로 다른 종류의 값(문자, 숫자, 문자열, 불리언 등)을 담을 수 있다.
    - 즉, 여러 Column으로 구성된 테이블 방식의 데이터를 표현하는 자료구조이다.
- 로우와 컬럼에 대한 색인을 가지고 있는데, 색인의 모양이 같은 Series 객체를 담고 있는 파이썬 사전으로 생각할 수 있다.
- 내부적으로 데이터는 리스트나 사전, 1차원 배열을 담고 있는 다른 컬렉션이 아니라, 하나 이상의 2차우 ㅓㄴ 배열에 저장된다.
- `pd.DataFrame(data)` 로 데이터프레임을 생성한다.
- 각 컬럼은 서로 다른 형태의 데이터를 표현할 수 있고, row와 column에 대한 인덱스를 가짐
    - 기본적으로 첫번째 칼럼은 0~N-1의 정수 인덱스 색인을 가진다.
    - `columns` 속성 : 칼럼의 위치를 직접 지정
    - `index` 속성 : 각 행의 인덱스를 직접 지정
- `pd.head()` : 첫 5개 행 데이터 출력

```python
def DataFrameFunc():
    # 각 칼럼의 색인과 데이터들
    data = {
        "state" : ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"], # state 칼럼
        "year" : [2000,2001,2002,2001,2002,2003], # year 칼럼
        "pop" : [1.5,1.7,3.6,2.4,2.9,3.2] # pop 칼럼
    }
    frame = pd.DataFrame(data) # data를 판다스의 데이터프레임 객체로 생성
    print(frame)
		print(frame.columns, frame.index)
    print(frame.head()) # 첫 5개 행의 정보를 추출
    

    # columns를 지정하면 데이터를 지정된 순서로 관리 가능
    frame2 = pd.DataFrame(data, columns=["year","state","pop"])
    print(frame2)
   
		# 직접 색인 인덱스의 이름 지정 가능
    frame3 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"],
                          index=["one", "two", "three", "four", "five","six"])
    print(frame3) # `debt` 에 대한 칼럼 데이터들은 결측치로 나타남
    
```

- 사전 표기법 또는 속성 값으로 데이터에 접근 가능
    
    ```python
        # 사전과 같은 표기법 또는 속성 값으로 특정 데이터에 접근 가능
        print(frame2["state"]) 
    		print(frame2.year) 
    ```
    
- 로우의 경우, `loc[]` 속성을 이용해 이름을 통해 접근 가능
    
    ```python
    		print(frame2.loc['three'])
    ```
    
- 배열에 직접 값을 할당 가능하다.
    
    ```python
        # 비어있는 칼럼에는 스칼라 값이나, 배열의 값이 할당 될 수 있음
        frame3["debt"] = np.arange(6)
        print(frame3)
        
        # 배열 값 할당
        val = pd.Series([-1.2, -1.5, -1.7], index=["two", "four", "five"]) # 지정한 인덱스에 대한 값만 할당
        frame3["debt"]=val 
        print(frame3)
    ```
    
- 새로운 칼럼 추가, 삭제 (`del` 예약어)
    
    ```python
    		# eastern 칼럼 추가 (Ohio state인 경우에만 True값 부여)
    		frame3["eastern"]=(frame3.state=="Ohio")
        print(frame3)
        
    		# 칼럼 삭제
        del frame3["debt"]
        print(frame3)
    ```
    

- 중첩된 사전을 이용해 데이터 생성
    - 바깡테 있는 사전의 키는 컬럼이 되고, 안에 있는 키는 로우가 된다.
    
    ```python
    pop = {'Nevada' : {2001:2.4, 2000:2.3},
    				'Ohio' : {2000:1.5, 2001: 3.4, 2002: 3.6}}
    frame3 = pd.Dataframe(pop)
    print(frame3) # Nevada와 Ohio가 칼럼, 2000/2001/2002가 로우가 됨
    ```
    
- `frame3.T` 를 이용해 전치 가능 (로우와 컬럼이 뒤집어짐)
- 직접 색인 지정 가능 : `pd.DataFrame(pop, index=[2001, 2002, 2003])`

<br>

## 색인 객체(Index Objects)

- Pandas 인덱스 객체는 테이블 형식의 데이터에서, 각 축에 대한 레이블과 메타 데이터를 다룸
- Series, Data Frame 객체 생성할 때 배열이네 레이블의 순서가 내부적으로 인덱스로 변환됨
- 파이썬의 사전과 달리 중복되는 인덱스 값을 허용함

```python
def PandasIndexes():
    # 인덱스 객체
    obj = pd.Series(range(3), index=['a', 'b', 'c'])
    print(obj, obj.index) 
    
		# 인덱스, 칼럼 포함 여부 확인
		print('a' in obj.columns) # True
		print(3 in obj.index) # False
```

<br>

# 2. 핵심 기능

- Series나 DataFrame에 저장된 데이터를 다루는 핵심적인 기능을 살펴보자.

### 재색인

- `reindex()` : 새로 재배치한 인덱스에 맞게 데이터을 재배열하고, 존재하지 않는 새로운 색인값에 대홰서는 NaN으로 채운다.

```python
    obj2 = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
    print(obj2)

		# 재색인(Reindex) : 인덱스의 배열을 변경 (없던 인덱스 추가 시 NaN 추가)
    obj3 = obj2.reindex(['a', 'b', 'c', 'd', 'e'])
    print(obj3)
```

- `method=ffill` 옵션 추가 :  누락된 값을 직전의 값으로 채워넣음
    - 시계열 같은 순차적인 데이터를 재색인할 때 값을 보간하거나 채워 넣어야 하는 경우에 사용됨

```python
    obj = pd.Series(['blue', 'purple', 'yellow'], index=[0,2,4])
    obj4 = obj.reindex(range(6), method='ffill') # index=1 은 0의 값으로, 3은 2의 값으로, 5는 4의 값으로 채움
    print(obj4)
```

- 재색인은 로우(색인), 컬럼 둘다 변경 가능 (재배치 기능)
    - `columns` 예약어를 통해 칼럼 재색인 가능

```python
    frame = pd.DataFrame(np.arange(9).reshape(3,3),
                         index=['a','b','c'],
                         columns=['Ohio', 'Texas', 'California'])
    print(frame)
    frame2 = frame.reindex(['a', 'b', 'c', 'd']) # d 칼럼 추가
    print(frame2)
    
    # columns 키워드를 사용해 DataFrame 컬럼 재색인 가능
    states = ['Texas','Utah','California'] # Ohio 컬럼은 제외, Utah를 새로 생성
    frame2 = frame.reindex(columns=states) # Utah에 대해서는 NaN으로 채움
    print(frame2)
```

- 재색인은 loc을 이용해 라벨로 색인하면 좀 더 간결하게 할 수 있다.

```python
frame.loc[['a', 'b', 'c', 'd'], states]
```

<br>


### 엔트리 삭제 (Dropping entries from an axis)

- 로우나 컬럼을 쉽게 삭제할 수 있다. 이는 데이터의 모양을 변경하는 작업을 진행한다.
- `drop` 메소드를 통해 축으로부터 지정된 값(칼럼)을 삭제한 새로운 객체 리턴
- 크기 또는 형태를 변경하는 함수는 새로운 객체 반환 대신 원본 객체를 변경한다.

```python
def DroppingEntries():
    obj = pd.Series(np.arange(5.), index=['a','b','c','d','e'])
    print(obj)
    
    new = obj.drop('c') # C 로우 삭제 (해당 로우의 데이터 모두 삭제)
    print(new)
    new = obj.drop(['d','c']) # 동시에 여러 로우 삭제
    print(new)
    
    data = pd.DataFrame(np.arange(16).reshape((4,4)),
                        index=['Ohio','Colorado','Utah','New York'],
                        columns=['one','two','three','four'])
    print(data)
    d1 = data.drop(['Colorado', 'Ohio'])
    print(d1)
```

- 컬럼의 값을 삭제할 때에는 `axis=1` 또는 `axis='columns'` 를 인자로 넘겨주면 된다.

<br>


## 인덱싱, 선택, 필터링 (Indexing, Selection, Filtering)

- Series 인덱싱은 NumPy 배열 인덱싱과 유사하게 동작하며, 인덱스가 정수가 아니어도 된다.
- 라벨을 이용한 슬라이싱의 경우, 시작점과 끝점을 모두 포함한다.

```python
import numpy as np
import pandas as pd

# Pandas의 Series데이터의 인덱싱, 셀렉션, 필터링
def indexing():
    obj = pd.Series(np.arange(4.), index=['a','b','c','d'])
    print(obj)
    print(obj['b']) # 직접 지정한 인덱스로 접근
    print(obj[2:4]) # 정수 인덱스로 이용한 슬라이싱
    print(obj[[1,3]]) # 1번, 3번 로우 데이터
    print(obj['b':'d']) # 인덱스 라벨 이름으로 슬라이싱
		print(obj[obj<2])
		obj['b':'c']=5 # 'b'와 'c'사이의 값(시작점, 끝점 포함)을 5로 변경
```

- 인덱스 또는 불리한 결과값을 이용해 로우 데이터에 접근이 가능하며, 스칼라  비교를 통해 불리언 DataFrame을 사용해 값을 선택할 수 있다.

```python
def indexing2():
    data = pd.DataFrame(np.arange(16).reshape((4,4)),
                        index=['Ohio', 'Colorado', 'Utah', 'New York'], # 행에 대한 인덱스 명시
                        columns=['one','two','three','four']) # 열에 대한 인덱스 명시
    print(data)
    print(data['two']) # two 인덱스에 대한 로우 데이터 출력
    print(data[:2]) # 0, 1번 로우 출력
		print(data[data['three']>5]) # boolean 결과값을 이용해 로우 데이터 출력하기
	  print(data<5) # 데이터가 5 미만이면 True, 아니면 False
		data[data<5] = 0
```

### loc과 iloc으로 설택하기

- `loc` 과 `iloc` 으로 행, 열에 매칭되는 데이터 값 접근하여 필터링하기
- 축 이름을 선택할 때에는 `loc` 을, 정수 색인으로 선택할 때는 `iloc` 을 사용한다.
    - loc은 시작점과 끝점을 모두 포함, iloc의 경우 끝점은 포함하지 않음을 주의해야 한다.

```python
    d = data.loc['Colorado', ['two','three']] # Colorado 행 데이터 중 two, three 열의 값 저장
    print(d)
    d = data.iloc[2,[3,0,1]] # 2행 데이터 중 3,0,1 열의 값 저장
    print(d)
		print(data.iloc[2]) # 2번 행 데이터 모두 출력
		print(data.iloc[[1,2],[3,0,1]] # 1,2번 행 중 3,0,1번 열 데이터 출력

		# 슬라이스, 라벨 리스트
		print(data.loc[:'Utah', 'two']) # Utah 로우까지, two 컬럼 데이터 출력
		print(data[:,:3][data.three>3]) # 모든 로우에 대해, 2번 컬럼까지, three의 값이 3 초과하는 것에 대해서만
```



<br>

## 산술연산과 데이터 정렬 (Arithmetic & Data Alignment)

- 서로 다른 인덱스를 가진 객체들 사이에서 산술 연산 수행이 가능
- 동일한 인덱스를 가진 값에 대해서 산술 연산 진행하며, 한쪽이라도 value가 없는 경우 NaN 결과값이 도출되어 색인을 통합하여 얻은 결과를 도출 (DB의 외부 조인과 유사)

```python
    s1 = pd.Series([7.3,-2.5,3.4,1.5], index=['a','c','d','e'])
    s2 = pd.Series([-2.1,3.6,-1.5,4,3.1], index=['a','c','e','f','g'])
    print(s1+s2)
    
    df1 = pd.DataFrame(np.arange(9.).reshape((3,3)),
                       columns=list('bcd'),
                       index=['Ohio','Texas','Colorado'])
    df2 = pd.DataFrame(np.arange(12.).reshape((4,3)),
                       columns=list('bde'),
                       index=['Utah','Ohio','Texas','Oregon'])
    
		print(df1+df2) # c, e 인덱스에 대해서는 모두 NaN 도출
```

```python
df = pd.DataFrame({'A': [1,2]})
df2 = pd.DataFrame({'B': [3,4]})

print(df1 - df2) # A,B 인덱스에 대해 모두 NaN 데이터로 도출
```

### 산술연산 메소드에 채워 넣을 값 지정하기

- 서로 다른 색인을 다니는 객체 간의 산술 연산에서 존재하지 않는 축의 값을 특정 값으로 지정하고자 하는 경우
    - 로우 인덱스와 컬럼 이름을 사용해 지정된 위치의 값 변경
    - 산술연산 메소드를 이용해 객체의 데이터 연산
    - `df.loc[1,'b'] = np.nan` → 1번 행, ‘b’번 열에 수정할 값을 할당

```python

    # 특정 값으로 채우기
    df1 = pd.DataFrame(np.arange(12.).reshape((3,4)),
                      columns=list('abcd'))
    df2 = pd.DataFrame(np.arange(20.).reshape((4,5)),
                       columns=list('abcde'))
        
    print(df1)
    print(df2)
    
    df2.loc[1,'b'] = np.nan # 1번 행, 'b'번 열의 데이터에 nan 값 넣기
    
    df = df1.add(df2, fill_value=0) # df1과 df2를 더하는데, 겹치지 않는 색인의 값에 대해서는 0으로 채워서 더하기
    print(df)

    print(1/df) # 나누어질 수 없는 경우는 inf
    print(df.rdiv(1)) # reverse한 후 분자를 1로 설정
```

- 재색인 할 때 fill_value 지정하기
    
    ```python
    df1.reindex(columns=df2.columns, fill_value=0) # df2의 columns로 재색인
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
    
		# 긱 로우에 대한 연산 (axis=0 옵션 또는 axis='index')
    series2 = frame['d'] # d번 열
    print(series2)
    
    # 'd'번 열의 모든 로우에 대해 각각 subtract 
    val = frame.sub(series2, axis='index')
    print(val)
```

### 함수 응용과 매핑 (Func. Application & Mapping)

- NumPy의 유니버셜 함수(원소 단위의 배열 처리 함수)와 Pandas 객체를 함께 동작
- 1차원 배열에 대한 함수를 각 컬럼과 로우에 적용하는 동작도 자주 활용됨 (람다 함수 매핑)
- `apply()` 메소드에 `axis=columns` 를 사용하면 주어진 함수를 각 칼럼마다 한번씩 적용 가능 , 기본적으로는 axis=0, axis=index

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
    print(v) # b, d, e 의 각 로우별 데이터들의 max-min
    
    v  = frame.apply(f, axis='columns') # 각 행의 데이터들 중 최대값 최소값에 대한 차이
    print(v) # utah, ohio, texas, oregon 의 각 칼럼에 대하여 max-min 
    
    # 원소 단위의 python 함수를 pandas와 함게 사용
    def f(x):
        return pd.Series([x.min(), x.max()], index=['min','max'])
    v = frame.apply(f)
    print(v)
```

### 정렬과 순위 (Sorting & Ranking)

- `sort_index()` : 로우나 컬럼의 색인을 알파벳 순으로 정렬
    - 기본적으로는 index에 대해 알파벳 순 오름차순 정렬
    - `axis=1` 옵션 → columns에 대해 정렬
    - `ascending=False` 옵션 → 내림차순 정렬

```python
def sorting():
    obj = pd.Series(range(4), index=['d','b','a','c'])
    print(obj)
    print(obj.sort_index()) # 인덱스를 기준으로 sort (인덱스 : a,b,c,d 순)
    
    frame = pd.DataFrame(np.arange(8).reshape((2,4)),
                         index=['three','one',],
                         columns=['d','a','b','c'])
    print(frame)
    print(frame.sort_index()) # index 알파벳 순으로 정렬
    print(frame.sort_index(axis=1)) # 칼럼에 대한 정렬 (abcd)
		
		# 내림 차순 정렬
		print(frame.sort_index(axis=1, ascending=False)) # 칼럼에 대한 내림차순 정렬 (dcba)
```

- `sort_values()` : 시리즈를 객체의 값에 따라 정렬

```python
obj = pd.Series([4,7,-3, np.nan, 2])
obj.sort_values() # -3, 2,4,7, np.nan  # 빈 값은 마지막에 위치
```

- `by` 옵션 → 하나 이상의 컬럼에 있는 값으로 정렬하는 경우 컬럼 이름 지정

```python
frame = pd.DataFrame({'b':[2,4,3,1], 'a':[3,-2,6,1]})
frame.sort_values(by='b') # b 칼럼을 정렬
frame.sort_values(by=['b','a']) # b 기준으로 먼저 정렬 후, 동일한 값에 대하여 a에 대해 정렬
```

- **************************순위(ranking)************************** : 1부터 배열의 유효한 데이터 개수까지 순서를 매긴다. Series와 DataFrame의 `rank()` 메서드는 동점인 항목에 대해서는 평균 순위를 매김
    - default : 4위에 해당하는 요소가 2개인 경우, 4,5를 부여해야 하므로 두 값의 랭킹에 동일하에 4.5, 4.5를 부여)
    - `method=first` 옵션 : 먼저 나타난 순서에 높은 순위 부여
    - `method=max` : 같은 값을 가지는 그룹을 높은 순위로 매김
    - `ascenidng=False` : 내림차순으로 랭킹 부여
    - `axis=columns` : index와 columns중 columns를 먼저 기준으로 하여 랭킹 부여

```python
# 1부터 오름차순으로 값을 부여
def ranking():
    obj = pd.Series([7,-5,7,4,2,0,4]) # 같은 값이 여러 개 있는 경우 평균값으로 랭크 값을 부여 (3,4번째인 경우 3.5, 3.5 부여)
    print(obj.rank()) # default 옵션 (average)
    
    # 동일한 값이 있을 때 먼저 마주치는 데이터에 랭크를 먼저 부여
    print(obj.rank(method='first'))
```


### 중복 색인

- 모든 Pandas함수들이 독립적인 레이블을 가지도록 하지만, 경우에 따라서는 중첩된 레이블(인덱스)를 가질 수 있음
    - `obj.index.unique` 속성을 이용해 레이블의 값이 유일한지 아닌지 알려줌
        - 중복되는 색인으로 접근했을 때 하나의 Series 객체를 반환함

```python
def sameLabel():
    obj = pd.Series(range(5), index=['a','a','b','b','c'])
    print(obj)
    print(obj.index.is_unique) # 모든 레이블이 unique한가? -> False
    print(obj['a']) # Series 객체 반환
```

<br>

# 3. 기술 통계 계산과 요약

- pandas 객체는 하나의 Series나 DataFrame의 로우나 컬럼에서 단일 값(합, 평균 등)을 구하는 ****축소**** 또는 ******************요약 통계****************** 범주에 속한다.
    - `sum()` : 열 단위 합계 (해당 열의 모든 로우 데이터를 더함)
        - `sum(axis=1)` , `sum(axis=columns)` : 행 단위 합계 (해당 행의 모든 컬럼 데이터를 더함)
    - `mean()` : 평균값
    - `skipna = False` 옵션 : 누락된 값을 제외하지 않음 (무조건 NaN 도출)
    - `idmax()` , `idmin()`: 최대/최소값을 갖는 인덱스 출력
    - `cumsum()` : 누적합
    - `describe()` 메서드 : 칼럼에 대해 통계 결과를 만들어줌


```python
def pandasStat():
    df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                       [np.nan, np.nan],[0.75, -1.3]],
                      index=['a','b','c','d'],
                      columns=['one','two'])
    print(df)
    print(df.sum()) # 열 단위 합계
    print(df.sum(axis='columns')) # 행 단위 합계
    print(df.mean(axis='columns',skipna=False)) # 행 단위 평균값 (NaN 인 경우 제외)
    print(df.idxmax()) # 열 단위 최댓값의 인덱스
    print(df.cumsum()) # 열 단위 누적합 계산
    print(df.describe()) # 칼럼에 대한 설명
```
