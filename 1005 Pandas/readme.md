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