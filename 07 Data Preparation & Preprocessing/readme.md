# CH7. 데이터 정제 및 준비

- `pandas` 라이브러리는 데이터를 원하는 형태로 가공하는 작업을 유연하고 빠른 고수준의 알고리즘과 처리 기능을 제공
- 결측치, 중복 데이터, 문자열 처리 그리고 다른 분석 데이터 변환에 대한 도구들을 다루어보자

<br>


## 1. 누락된 데이터 처리하기 (Handling Missing Data)

- 누락 데이터를 처리하는 일은 데이터 분석 애플리케이션에서 흔히 발생하는 일이며, pandas의 설계 목표 중 하나는 누락 데이터를 가능한 한 쉽게 처리할 수 있도록 하는 것이다.
- 산술 데이터의 경우 pandas는 누락된 값을 쉽게 찾을 수 있도록 하기 위해 누락된 데이터를 실숫값인 NaN으로 취급한다.
- 분석을 위해 데이터를 정제하는 과정에서 결측치 자체를 데이터 수집 과정에서의 실수나 결측치로 인한 잠재적인 편향을 찾아내는 수단으로 인식하는 것은 매우 중요
- 산술 데이터에 한해 pandas는 누락 데이터를 실숫값인 NaN으로 취급한다.
    - `numpy.nan` 또는 파이썬 내장 `None` 값을 누락값으로 취급
    
    ```python
    def find_null_data():
        # 누락값 np.nan은 NaN으로 표시 
        string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
        print(string_data)
        print(string_data.isnull())
        
        # 파이썬의 내장 None 값 또한 NaN으로 인식
        string_data[0] = None
        print(string_data.isnull())
    ```

<br>    

### NA 처리 메서드

- `dropna()` : 누락된 데이터가 있는 축(로우, 컬럼)을 제외시팀킴 - 어느 정도의 누락 데이터까지 용인할 것인지 지정 가능
- `fillna()` : 누락된 데이터를 대신할 값을 채우거나 ‘ffill’, ‘bfill’ 같은 보간 메서드를 적용
- `isnull()` : 누락되거나 NA인 값을 알려주는 불리언 값이 저장된 같은 형의 객체를 반환
- `notnull()` : isnull과 반대되는 메서드

<br>

### 누락된 데이터 골라내기

- `dropna()` 매서드를 이용해 null이 아닌 데이터와 색인값만 들어있는 Series 반환해보자
    - `how='all'` 인자는 전체가 NA 값인 로우만 제외
    - `axis=1` 인자는 컬럼에 적용
    
    ```python
    # dropna 메서드로 결측치 로우 제외하기
    def dropna_method():
        data = pd.Series([1, NA, 3.5, NA, 7])
        # not null인 Series만 반환
        print(data.dropna())
        print(data[data.notnull()])
        
        # 2차원 DataFrame에 dropna() 적용
        data = pd.DataFrame([[1., 6.5, 3.],
                             [1., NA, NA],
                             [NA, NA, NA],
                             [NA, 6.5, 3.]])
        clenaed = data.dropna() # 하나라도 NA를 포함하고 있는 축은 제외
        print(cleaned)
        
        # 모두 NA 값인 로우만 제외
        cleaned_all_none = data.dropna(how='all') 
        
        data[4] = NA
        print(data)
        
        # 전체가 NA인 컬럼 제외하기
        cleaned_column = data.dropna(axis=1, how='all')
        print(cleaned_column)
    ```
    
- 시계열 데이터의 경우, 몇 개 이상의 값이 들어있는 로우만 살펴보고 싶은 경우 `thresh` 인자 사용하기
    
    ```python
    # 값이 특정 개수 이상 들어있는 로우만 선택  
    def dropna_threshold():
        df = pd.DataFrame(np.random.randn(7,3))
        df.iloc[:4,1]=NA
        df.iloc[:2,2]=NA
        print(df)
        print(df.dropna())
      
        # thresh를 2로 설정하여 2개 이상의 값이 들어있는 로우에 대해서만 DF 객체 전달
        print(df.dropna(thresh=2))
    ```
    
<br>


### 결측치 채우기

- dropna와 같이 누락된 값을 제외시키지 않고, 데이터 상의 구멍을 메꾸기
- `fillna()`  메서드 사용
    - 사전 값으로 각 컬럼마다 다른 값을 채울 수도 있음
    - `inplace=True` 인자로 기존 객체를 변경 가능
    - `method='ffill'` 인자로 재색인, 또는 `limit` 인자로 최대 재색인 허용 값 지정 가능
    
    ```python
    # fillna로 값 채우기
    def fillna_method():
        df = pd.DataFrame(np.random.randn(7,3))
        print(df.fillna(0)) # 누락값을 0으로 채우기
        
        # 사전값을 넘겨서 각 컬럼마다 다른 값을 채우기
        print(df.fillna({1:0.5, 2:0})) # 1번 컬럼의 NA값에는 0.5, 2번 컬럼의 NA값에는 0
        
        # inplace 인자로 기존 객체를 변경
        _ = df.fillna(0, inplace=True)
        print(df)
        
        # 재색인에서 사용 가능한 보간 메서드 적용
        df = pd.DataFrame(np.random.randn(6,3))
        df.iloc[2:,1]=NA
        df.iloc[4:,2]=NA
        print(df)
        
        # NA 값은 컬럼의 이전 값으로 대체
        print(df.fillna(method='ffill'))
        print(df.fillna(method='ffill', limit=2)) # 최대 2번까지 ffill 허용
        
        # Series의 평균값 전달하여 채우기 가능
        data = pd.Series([1., NA, 3.5, NA, 7])
        print(data.fillna(data.mean()))
    ```
    

<br>

## 2. 데이터 변형 (Data Transformation)

### 중복 제거하기

- `df.duplicated()`  : 중복된 로우 발견하기
- `df.drop_duplicates()` : duplicated 배열이 False인 DataFrame 반환
- 기본적으로 두 메서드는 모든 컬럼에 적용되며, 처음 발견된 값을 발견하거나 반환
    - `keep='last'` 옵션은 마지막으로 발견된 값을 반환
    - 중복 찾아내기 위한 부분합 따로 지정 가능
    
    ```python
    def duplicated_method():
        data = pd.DataFrame({'k1':['one', 'two']*3 + ['two'],
                             'k2':[1,1,2,3,3,4,4]})
        print(data)
        print(data.duplicated()) # 각 로우가 중복인지 아닌지 불리언 Series 반환
        
        print(data.drop_duplicates()) # duplicated 배열이 False인 DataFrame 반환
        
        data['v1'] = range(7)
        
        # 중복을 찾아내기 위한 부분합을 따로 지정하기
        print(data.drop_duplicates(['k1'])) # k1 컬럼에 기반하여 중복 걸러내기
        
        # a마지막으로 발견된 값을 반환
        print(data.drop_duplicates(['k1', 'k2'], keep='last'))
    ```
    

<br>

### 함수나 매핑을 이용해 데이터 변형하기

- `map()` 메서드를 이용해 데이터의 카테고리를 알려주는 칼럼 추가 가능
    - `str.lower()` : 사전류의 객체의 경우 대소문자 섞여있는 경우 소문자로 모두 변환하여 해결 가능
    - `map()` : 사전 객체의 key를 value값으로 변환

```python
# 데이터 형태 사전을 이용해 변환하기
def data_transformation_by_mapping():
    # 수집한 육류 데이터
    data = pd.DataFrame({'food': ['bacon','pulled pork','bacon', 'Pastrami','corned beef',
                                  'Bacon','pastrami','honey ham','nova lox'],
                         'ounces':[4,3,12,6,7.5,8,3,5,6]})
    print(data)
    
    # 육류가 어떤 동물의 고기인지 알려주는 컬럼 추가
    meat_to_animal = {
        'bacon':'pig',
        'pulled pork':'pig',
        'pastrami':'cow',
        'corned beef':'cow',
        'honey ham':'pig',
        'nova lox':'salmon'
    }
    
    # 소문자로 변경하기 (str.lower())
    lowercased = data['food'].str.lower()
    print(lowercased)
    # 각 육류의 고기 종류 컬럼 추가하기
    data['animal'] = lowercased.map(meat_to_animal)
    print(data)
    
    # 함수 넘겨 주는 방법
    data['food'].map(lambda x:meat_to_animal[x.lower()])
    

<br>

def main():
```

### 값 치환하기

- `replace()` 메서드와 사전, 리스트 조합을 이용하여 치환하려는 값을 지정한 값으로 바꾸기

```python
# 값 치환하기
de
f data_replacement():
    data = pd.Series([1., -999., 2., -999., -1000., 3.])
    print(data)
    
    # 누락된 값을 나타내는 -999를 pandas에서 인식할 수 있는 NA 값으로 치환하기
    data.replace(-999, np.nan)
    data.replace([-999, -1000], np.nan) # 한 번에 여러 값 치환하기
    data.replace([-999, -1000], [np.nan, 0]) # 치환하는 값마다 다른 값으로 치환하기
    data.replace({-999: np.nan, -1000:0}) # 사전을 이용하기 가능
```

<br>

### 축 색인 이름 바꾸기

- `Dataframe.index.map()` 을 이용해 인덱스 색인 이름 변경하기
- `rename()` : 원래 객체를 변경하지 않고 새로운 객체를 생성하는 경우
    - index = str.title, columns=str.upper 과 같이 인자 옵션 지정 가능
- `inplace=True` 옵션으로 원본 데이터 변경

```python
# 축 색인 이름 바꾸기
def change_column_name():
    data = pd.DataFrame(np.arange(12).reshape((3,4)),
                        index=['Ohio','Colorado','New York'],
                        columns=['one','two','three','four'])
    transform = lambda x:x[:4].upper()
    print(data.index.map(transform)) # map 이용해 축 색인 바꾸기
    
    # 대문자로 변경된 축 이름을 바로 대입
    data.index = data.index.map(transform)
    print(data)
    
    # 원래 객체 변경하지 않고 새로운 객체 생성하는 경우: rename()
    print(data.rename(index=str.title, columns=str.upper))
    print(data.rename(index={'OHIO':'INDIANA',},
                      columns={'three':'peekaboo'})) # 사전 객체를 이용해 일부 축 이름 변경
    # 원본 데이터 변경 시
    data.rename(index={'OHIO':'INDIANA'}, inplace=True)
    print(data)
```

<br>

### 이산화(개별화)와 양자화

- 개별화: 연속성 데이터를 개별로 분할하거나 분석을 위해 그룹별로 나누는 경우 필요
    - `pandas.cut(a,b)` : a리스트를 b리스트의 그룹을 기준으로 나누어 그룹화하여 Categorical 객체 반환
        - codes : 속한 그룹 (0부터 시작) 의 번호
        - categories: 카테고리 이름
        - `value_counts(cutted)` : Categorical 객체 cutted를 각 그룹별로 속한 데이터 개수 센 Series 반환
        - b 자리에 그룹 리스트가 아닌 정수 값을 넘겨주는 경우, 해당 정수만큼  균등한 길이의 그룹을 자동으로 계산
    - `pandas.qcut()` : 표본 변위치(샘플 수)를 기반으로 데이터를 분할
        - cut() 의 경우 데이터의 분산에 따라 각각 그룹마다 데이터 수가 다르게 나뉘는데, qcut() 은 표준 변위치를 사용하여 적다히 같은 크기의 그룹으로 나눌 수 있음
    
    ```python
    # 그룹 분석    
    def discretization():
        ages = [20,22,25,27,21,34,37,31,61,45,41,32]
        bins = [18,25,35,60,100] # 18-25, 26-35, 36-60, 60 이상의 그룹
        # Catagoricals 객체 반환 : pandas의 cats는 각 ages 데이터가 속한 bins 그룹의 리스트 반환
        # 중괄호 쪽의 값은 포함하지 않고 대괄호 쪽의 값을 포함
        cats = pd.cut(ages,bins) 
        print(cats)
        
        print(cats.codes)
        print(cats.categories) # 카테고리의 이름 
        print(pd.value_counts(cats)) # 각 카테고리에 속한 데이터의 개수 반환
        
        # 중괄호 대신 대괄호 쪽이 포함되지 않도록 변경
        print(pd.cut(ages, [18,26,36,61,100], right=False))
        
        # labels 옵션으로 그룹의 이름 지정하기
        group_names = ['Youth','YoungAdult','MiddleAged', 'Senior']
        print(pd.cut(ages, bins, labels=group_names))
        
        
        # 그룹의 경계값이 아닌 그룹의 개수를 넘겨주면 데이터의 최솟값, 최댓값을 기준으로 균등한 길이의 그룹을 자동 계산
        data = np.random.randn(20)
        print(pd.cut(data, 4, precision=2)) # 균등분포 내에서 4개의 그룹으로 나누는 경우 (precision: 소수점 아래 2자리로 제한)
    
        # qcut: 표본 변위치를 기반으로 데이터 분할
        data = np.random.randn(1000)
        cats = pd.qcut(data, 4) # 4분위로 분류
        print(cats)
        
        print(pd.value_counts(cats)) # 각 그룹의 개수 250개
        
        # 변위치를 직접 지정 (0부터 1 사이)
        print(pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]))
    ```
    

<br>

### 특잇값을 찾고 제외하기

- 배열 연산 시 특잇값(outlier)을 제외하거나 적당한 값으로 대체하는 것이 중요
    - `np.sign(data)` : 양, 음수에 따라 1과 -1 반환
    
    ```python
    # 특잇값 outlier 제외하거나 대체하기
    def handle_outlier():
        data = pd.DataFrame(np.random.randn(1000,4)) # 4개의 컬럼에 각각 1000개의 실수 데이터 생성
        print(data.describe())
        
        # 한 컬럼에서 절댓값이 3을 초과하는 데이터 탐색
        col = data[2]
        print(col[np.abs(col)>3])
        print(data[(np.abs(data)>3)].any()) # 절댓값 3 초과하는 값이 들어있는 모든 로우 선택시
        
        data[np.abs(data)>3] = np.sign(data)*3 # 초과하는 값을 3, -3으로 변경
        print(data.describe())
        
        print(np.sign(data).head()) # 양수, 음수에 따라 1이나 -1 담긴 배열을 반환
    ```
    

<br>

### 치환과 임의 샘플링

- `np.random.permutation(n)` : 0부터 (n-1)까지의 값을 랜덤으로 재배치
- `DataFrame.take(sampler)` : sampler를 이용해 데이터프레임의 축 순서를 변경
- `DataFrame.sample(n=)` : 치환없이 임의로 n개의 축 선택하기
    - `replace=True` 옵션으로 반복 허용
    
    ```python
    # 치환과 임의 샘플링
    def permutatation_sampling_replacement():
        df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
        print(df)
        
        sampler = np.random.permutation(5) # 0부터 4까지 숫자를 임의로 재배치
        print(sampler)
        
        # sampler에 의해 take() 메서드로 배치 변경
        print(df.take(sampler))
        
        # sample()로 치환 없이 일부만 임의로 선택 
        print(df.sample(n=3)) # 3개의 랜덤 로우 선택
        
        # 반복 선택을 허용하여 표본을 치환을 통해 생성해 내기 (replace=True 옵션 지정)
        choices = pd.Series([5,7,-1,6,4])
        draws = choices.sample(n=10, replace=True) # choices 배열의 값으로 반복 허용하여 10개의 샘플 생성
        print(draws)
    ```

<br>   

### 표시자/더미 변수 계산하기

- 분류값을 ‘더미’나 ‘표시자’ 행렬로 전환하여 데이터를 변환해보자.
- `pandas.get_dummies()` : 어떤 DataFrame의 한 컬럼에 k가지의 값이 있는 경우 k개의 컬럼이 있는 DataFrame이나 행렬을 만들고, 값으로는 1과 0으로 채워넣기
    - `prefix` 인자 옵션 → 접두어 추가 가능
    - `join()` : 다른 데이터와 병합하기
    
    ```python
    # 표시자/더미 변수 계산하기
    def dummies():
        df = pd.DataFrame({'key':['b','b','a','c','a','b'],
                           'data1':range(6)})
        print(pd.get_dummies(df['key'])) # key의 값이 컬럼이 되고, 해당되는 값에 1, 아닌 경우 0을 지정
        
        # 컬럼에 접두어(prefix) 추가하고 다른 데이터와 병합하기
        dummies = pd.get_dummies(df['key'], prefix='key') # 컬럼이 key_a, key_b, key_c로 변경
        df_with_dummy = df[['data1']].join(dummies) # data1과 dummies 병합
        print(df_with_dummy)
    ```
    

<br>

### 영화 장르 데이터 다루기

```python
# movie 데이터의 genre를 카테고리로 개별화하기   
def movies_with_categories():
    mnames = ['movie_id','title','genres']
    movies = pd.read_table('datasets/movielens/movies.dat',sep='::', header=None, names=mnames,
                           engine='python', encoding='ISO-8859-1')
    print(movies[:10]) # 컬럼: movie_id, title, genres
    
    # 각 장르마다 표시자 값 추가하기
    all_genres = [] # 모든 장르 리스트
    for x in movies.genres:
        all_genres.extend(x.split('|'))
    genres = pd.unique(all_genres)
    print(genres)
    
    # 표시자 DataFrame 생성
    zero_matrix = np.zeros((len(movies), len(genres)))
    dummies = pd.DataFrame(zero_matrix, columns=genres)
    
    # 각 영화 순회하며 dummies의 각 로우 항목을 1로 설정
    gen = movies.genres[0]
    print(gen.split('|'))
    dummies.columns.get_indexer(gen.split('|'))
    
    for i, gen in enumerate(movies.genres):
        indices = dummies.columns.get_indexer(gen.split('|'))
        dummies.iloc[i, indices] = 1
         
    # movies와 조합하기
    movies_windic = movies.join(dummies.add_prefix('Genre_')) # Prefix 지정
    print(movies_windic.iloc[0])
```

## 3. 문자열 다루기

- 문자열 객체의 내장 메서드로 문자열 텍스트 처리 가능
- 복잡한 패턴 매칭이나 텍스트 조작 → 정규 표현식 이용
- pandas는 배열 데이터 전체에 쉽게 정규 표현식으로 적용하고, 누락된 데이터를 편리하게 처리할 수 있는 기능을 포함하고 있음

    <img src="https://user-images.githubusercontent.com/71310074/204183987-016beac0-f96d-40f5-a814-d9f446feb970.png" width=500>

    <img src="https://user-images.githubusercontent.com/71310074/204184017-fc8d3986-6472-434c-b257-958402a3f610.png" width=500>

### 문자열 객체 메서드

- `split(',')` : 특정 문자를 기준으로 문자열 자르기
- `strip()` : 앞뒤의 공백 문자 제거
- `':'join()` : 리스트를 어떠한 문자열로 연결하여 합치기
- `index(',')` : 문자열에서 특정 문자열 찾기 (찾지 못한 경우 예외 처리)
- `find(',')` : 문자열에서 특정 문자열 찾기 (찾지 못한 경우 -1 반환)
- `count(',')` : 특정 문자열의 발견 횟수 반환
- `replace('','')` : 찾아낸 패턴을 다른 문자열로 치환

```python
# 문자열 객체 메서드
def string_method():
    # split() - 특정 문자를 기준으로 문자열 자르기
    val = 'a,b, guido'
    print(val.split(',')) # 리스트 반환
    
    # strip() - 공백 문자 제거
    pieces = [x.strip() for x in val.split(',')] # 공백문자 제거한 문자열의 리스트 
    print(pieces)
    
    # ''.join() - 리스트를 문자열로 합치기
    print('::'.join(pieces))
    
    # 문자열 내 특정 문자열의 위치 찾기
    print('guido' in val)
    print(val.index(',')) # 처음으로 찾은 문자의 위치 찾기
    # print(val.index(':')) # 문자 찾지 못한 경우 예외 처리 발생
    print(val.find(':')) # 문자 찾지 못한 경우 -1 반환
    
    # count() - 특정 부분 문자열 발견 횟수 반환
    print(val.count(','))
    
    # replace() - 찾아낸 패턴을 다른 문자열로 치환
    print(val.replace(',', '::'))
```

### 정규 표현식

- **정규 표현식** : 텍스트에서 문자열 패턴을 찾는 유연한 방법을 제공
- regex 단일 표현식: 정규 표현 언어로 구성된 문자열 (파이썬의 `re` 모듈 내장)

    <img src="https://user-images.githubusercontent.com/71310074/204183814-62b3aed5-d1df-45a8-a648-6365da08b7cb.png" width=600>

- `re.split()` : 정규 표현식 컴파일 후 split메서드 실행
- `re.complie()` : 직접 정규 표현식을 컴파일하여 정규 표현식 얻기
- `re.findall()` : 정규 표현식에 매칭되는 모든 패턴의 목록 얻기
    
    ```python
    # 정규 표현식 활용하기
    def regular_expression():
        import re 
        
        #하나 이상의 공백 문자를 의미하는 \s+를 사용하여 문자열 분리
        text = "foo bar\t baz \tqux"
        print(re.split('\s+', text)) # 정규 표현식 컴파일 후 split 메서드 실행
        
        # 직접 정규 표현식을 컴파일하여 얻은 정규 표현식 객체를 재사용하기
        regex = re.compile('\s+')
        print(regex.split(text))
        
        # 정규 표현식에 매칭되는 모든 패턴의 목록 얻기
        print(regex.findall(text))
    ```
    

- `findall()` : 문자열에서 일치하는 모든 부분 문자열 찾기
- `search()` : 패턴과 일치하는 첫번째 존재 반환
- `match()` : 문자열의 시작 부분에서 일치하는 것만 찾음
    
    ```python
    # 이메일 예제로 정규 표현식 이해하기
    def regular_expression_email():
        import re 
        
        text = """Dave dave@google.com
        Steve steve@gmail.com
        Rob rob@gmail.com
        Ryan ryan@yahoo.com    
        """
        
        pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
        
        regex = re.compile(pattern, flags=re.IGNORECASE) # 대소문자 구분 안하기
        
        print(regex.findall(text)) # pattern에 해당하는 부분 문자열 리스트
        
        # search: 패턴과 일치하는 첫 번째 이메일 주소만 찾기
        m = regex.search(text)
        print(m) # 정규 표현 패턴이 위치하는 시작점, 끝점 반환
        print(text[m.start():m.end()])
        
        # match: 문자열의 시작점에서부터 일치하는지 검사
        print(regex.match(text))
        
        # sub: 주어진 문자열로 치환
        print(regex.sub('REDACTED', text))
        
        # 사용자 이름, 도메인 이름, 도메인 접미사 3가지 컴포넌트로 나누기
        pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
        regex = re.compile(pattern, flags=re.IGNORECASE)
        # match: groups메서드로 각 패턴 컴포넌트의 튜플 얻기
        m = regex.match('wesm@bright.net')
        print(m.groups())
        
        print(regex.findall(text))
        
        # \1, \2 같은 특수 기호로 각 패턴 그룹에 접근
        print(regex.sub(r'Username: \1 Doimain: \2, Suffix: \3', text))
    ```
    
    <img src = "https://user-images.githubusercontent.com/71310074/204183689-80478d5e-ed00-4e02-a172-35bd76bfaf02.png" width=600>

### pandas의 벡터화된 문자열 함수

- 문자열과 정규 표현식 메서드는 `[data.map](http://data.map)` 을 사용해 각 값에 적용할 수 있지만 NA 값을 만나면 실패함
- 이를 대처하기 위해 Series에서 str 속성을 이용해 NA 값을 건너뛰도록 한다.
- `str.contains()` : 어떤 문자열을 포함하고 있는지 검사
- `str.get()` , `str[0]` 와 같이 색인을 이용해 벡터화된 요소를 꺼내오기

```python
# 벡터화된 문자열 함수
def vectorized_string_method():
    # 누락된 값을 포함하는 데이터
    data = {'Dave':'dave@google.com',
            'Steve':'steve@gamil.com',
            'Rob':'rob@gmail.com',
            'Wes':np.nan}
    data = pd.Series(data)
    print(data)
    print(data.isnull())
    
    # NA 값을 넘어뛰도록 문자열의 gmail 포함 여부 확인
    print(data.str.contains('gmail'))
    
    pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
    print(data.str.findall(pattern, flags=re.IGNORECASE))
    
    matches = data.str.match(pattern, flags=re.IGNORECASE)
    print(matches)
    
    # print(matches.str.get(1))
    # print(matches.str[0])
    print(data.str[:5])
```