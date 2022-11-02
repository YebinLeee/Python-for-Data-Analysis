# CH 6. 데이터 로딩과 저장, 파일 형식

# 1.  텍스트 파일에서 데이터를 읽고 쓰는 법

- pandas에서 표 형식의 텍스트 데이터를 DataFrame 객체로 읽어오는 몇 가지 기능
    - ******옵션******
        - 색인 : 반환하는 DataFrame에서 하나 이상의 컬럼을 색인으로 지정. 파일이나 사용자로부터 컬럼 이름을 받거나 아무것도 받지 않을 수 있다.
        - 자료형 추론과 데이터 변환 : 사용자 정의 값 변환과 비어있는 값을 위한 사용자 리스트를 포함
        - 날짜 분석 : 여러 컬럼에 걸쳐 있는 날짜와 시간 정보를 하나의 컬럼에 조합해서 결과 반영
        - 반복: 여러 파일에 걸쳐 있는 자료를 반복적으로 읽어옴
        - 정제되지 않은 데이터 처리: 로우나 꼬리말, 주석 건너뛰기 또는 천 단위마다 쉼표로 구분된 숫자 같은 사소한 것들의 처리

### 데이터 처리 함수

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FAvezV%2FbtrPS0C1wc9%2F69Y9d4poc7KL4z4kKpuUY0%2Fimg.png" width="400">

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcJwaq4%2FbtrPVDtKTlg%2FD7wv38xg76lW5GN2gKozU1%2Fimg.png" width="400">

## 데이터 읽어 오기(Data Loading)

- 타입 추론(Type Inference)
    - `pandas.read_csv` 의 경우 데이터 형식에 자료형을 명시하지 않으므로 타입 추론을 수행
    
    ```python
    # CSV 파일들 열어 출력해보기
    def read_csv_file():
        for i in range(1,8):
            path = 'examples/ex{0}.csv'.format(i)
            df = pd.read_csv(path)
            print(df, end='\n\n\n') # 자동으로 index값 부여 (column이름은 header)
    ```
    
- 쉼표로 구분되어 있는 파일의 경우 `read_csv` 사용, 또는 `read_table`에 구분자를 쉼표로 지정해서 읽어올 수도 있음
    
    ```python
    df=pd.read_table(path, sep=',') # read_table의 구분자를 쉼표로 지정
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbojWhe%2FbtrPTlf4262%2FhVl0avQBUVOjzVRY1zQ831%2Fimg.png)
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbzj7R1%2FbtrPQqCKxj9%2FEY9DM6lYbjQLiIORLDV7K1%2Fimg.png)
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcPoO1Y%2FbtrPQxofUD0%2FkFC1IFnaA0EcaTgAeGXf0k%2Fimg.png)
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdtJLhj%2FbtrPQL7Gmix%2FTFE0QGE2NprxrZrvhET1x1%2Fimg.png)
    

### 컬럼 지정하기

- 컬럼 이름 자동으로 생성 또는 직접 지정 가능
    
    ```python
    # 컬럼 이름 자동 생성 (0~)
    df = pd.read_csv('examples/ex2.csv', header=None)
    
    # 컬럼 이름 직접 지정
    df = pd.read_csv('examples/ex2.csv', names=['a','b','c','d','message'])
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdEEOvc%2FbtrPQ3tr7kl%2FmSULgEKkkPBxcUChbK5QTk%2Fimg.png)
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnXrOb%2FbtrPT2UVd3w%2F8qQsJipPxkAhdZLDUgSlXk%2Fimg.png)
    

- message 컬럼을 색인으로 하는 데이터프레임 반환 → `index_col` 인자에 4번째 컬럼 또는 ‘message’이름을 가진 컬럼을 지정해서 색인으로 변경
    
    ```python
    names=['a','b','c','d','message']
    df = pd.read_csv('examples/ex2.csv', names=names, index_col='message')
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F5zOb1%2FbtrPWVOwHzr%2F5d8PQIlDv9dv0r4jF1cOG1%2Fimg.png)
    

### 계층적 색인

- 계층적 색인을 지정하기 (다중 컬럼으로부터 계층적 인덱스 설정하기)
    - key1, key2, value1, value2 컬럼들 중 key1과 key2를 인덱스로 설정
    
    ```python
    def csv_index():
        print(pd.read_csv('examples/csv_mindex.csv'))
        # 다중 컬럼으로부터 계층적 인덱스 설정하기
        parsed = pd.read_csv('examples/csv_mindex.csv', index_col=['key1','key2'])
        print(parsed)
    ```
    
    ![Untitled](https://blog.kakaocdn.net/dn/TRucx/btrPQIXo0R5/nlWFXk2xo5jj3hpkyHT6j1/img.png)
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F90fqW%2FbtrPRcjCje0%2FWxaRgvQKGPB5KCTP4MyPRk%2Fimg.png)
    

- 고정된 구분자가 없는 경우
    - 데이터의 로우보다 컬럼이 하나 적기 때문에 `read_table` 함수는 첫번째 열이 DataFrame의 객체의 색인(인덱스)라고 추론
    - 필드가 여러 개의 공백 문자로 구분되어 있으므로, 정규 표현식 `\s+` (whitespace 공백문자 기준으로 자르기) 를 사용하여 처리
    
    ```python
    # read_table을 이용해 인덱스 추론, 공백 문자를 기준으로 데이터값들 자르기
    def re_seperator():
        print(list(open('examples/ex3.txt')))
        result = pd.read_table('examples/ex3.txt',sep='\s+')
        print(result)
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FVYE6A%2FbtrPVDHjXhy%2F16AvFkX1w1109ccTqnvKcK%2Fimg.png)
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQ1q3Y%2FbtrPR6cpj8M%2F0SdavTYYkxERmcm6sGizL0%2Fimg.png)
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcjSkm1%2FbtrP0fZ04I0%2FIuky1RhzxYM8yx41uy1SK1%2Fimg.png)
    

- `skiprows` 옵션을 이용해 선택된 로우 제외하기
    
    ```python
    # 로우 제외하기
    def skip_rows():
        orig = pd.read_csv('examples/ex4.csv')
        print(orig)
        ret = pd.read_csv('examples/ex4.csv', skiprows=[0,2,3]) # 0,2,3번 로우 건너뛰기
        print(ret)
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F6Rm0e%2FbtrPQxBStJY%2FkV2CKLksOXZNZZE1w6cdBK%2Fimg.png)
    

### 누락된 값 처리하기

- 텍스트 파일에서 누락된 값은 표기되지 않거나(빈 문자열), 구분하기 쉬 운 특수한 문자로 표기됨 (pandas는 보통 NA나 NULL과 같은 통용되는 문자를 비어잆는 값으로 사용)
    
    ```python
    # 손실 데이터 처리하기(빈 문자열 NaN)
    def missing_data():
        ret = pd.read_csv('examples/ex5.csv')
        print(ret, end='\n\n') # 결측치를 포함하는 파일 (NaN 표시)
        print(pd.isnull(ret), end='\n\n') # 불린으로 null값 여부 확인
        
        # na_values 옵션으로 손실값 처리
        result = pd.read_csv('examples/ex5.csv', na_values=['NULL'])
        print(result, end='\n\n')
        
        # 컬럼마다 다른 NA문자를 사전값으로 넘겨 처리 (message 컬럼의 foo와 NA, something 컬럼의 two)
        sentinels = {'message':['foo','NA'], 'something':['two']}
        print(pd.read_csv('examples/ex5.csv', na_values=sentinels))
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdadHqz%2FbtrP0g5G7t7%2FNCkJa9W2AYWCD5GY1pQaB0%2Fimg.png)
    

### read_csv, read_table 함수의 옵션

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbl8djL%2FbtrPQIJUNIf%2F4jHYy8zWD5M65ZQy5Ft2lK%2Fimg.png)

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcTbP0F%2FbtrPQNxGlm1%2Fgkyt5KF4e2TXncaze97ZlK%2Fimg.png)

### 텍스트 파일 조금씩 읽어오기

- 매우 큰 파일을 처리할 때 인자를 제대로 주었는지 알아보기 위해, 파일의 일부분만 읽어보거나 여러 파일 중 몇 개의 파일만 읽어서 확인할 경우가 있다.
- 큰 파일을 다루기 전 pandas의 출력 설정을 다음과 같이 정할 수 있다.
    
    ```python
    def read_parts():
        pd.options.display.max_rows=10 # 최대 10개의 데이터만 출력
        result = pd.read_csv('examples/ex6.csv')
        print(result, end='\n\n')
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbBI41K%2FbtrPUmZ8hsQ%2Fk4G9D3MYIQjJUkhki2xTck%2Fimg.png)
    

- `nrows` 옵션으로 설정하여 파일에서 지정된 로우만큼 읽어오기
    
    ```python
    # nrows 옵션 설정하기
    print(pd.read_csv('examples/ex6.csv', nrows=5))
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FZ4B07%2FbtrP0gR97oD%2F3pLwsbrl3rY2Nc6XEKZq1K%2Fimg.png)
    

- 로우의 줄 수를 청크 크기로 지정하고, 청크 단위로 읽어 오기
    - `chunksize` 을 지정하여 read_csv에서 반환된 TextParser 객체를 이용해 chunksize에 따라 분리된 파일들을 순회할 수 있다.
    - ex6.csv 파일을 순회하며 ‘key’ 로우에 있는 값을 세어보자.
    
    ```python
    # 파일을 여러 조각으로 나누어 읽기
    def read_chunk():
        chunker = pd.read_csv('examples/ex6.csv', chunksize=1000) # chunksize를 1000으로 설정 
        print(chunker) # TextParser 객체 반환
        
        # Series 객체 tot에 chuncker의 요소들의 'key' 로우에 있는 값 세어보기
        tot = pd.Series([])
        for piece in chunker:
            tot = tot.add(piece['key'].value_counts(), fill_value=0)
        
        # tot을 내림차순으로 정렬    
        tot = tot.sort_values(ascending=False)
        print(tot[:10])
    ```
    
    ![Untitled](https://blog.kakaocdn.net/dn/brEqED/btrPQrO9SDR/sYnLuz2b6nOtMkl8lDTaS0/img.png)
    

## 데이터를 텍스트 형식으로 기록하기 (Data Export)

- 데이터를 구분자로 구분한 형식으로 내보내기
- `sep` 옵션으로 구분자 지정하기
- `na_rep` 옵션으로 누락된 값 표시할 문자열 지정하기
- `index=False`, `header=False` 옵션 지정하여 로우, 컬럼 이름을 포함하지 않을 수 있음
- `columns` 옵션으로 컬럼의 일부분만 기록하고 순서를 지정 가능
    
    ```python
    # csv 파일로 저장
    def to_csv_export():
        data = pd.read_csv('examples/ex5.csv')
        print(data, end='\n\n')
        
        # to_csv 메서드 이용하여 데이터를 쉼표로 구분된 형식으로 파일에 쓰기
        data.to_csv('examples/out.csv')
         
        # 파일에 기록하지 않고 sys.stdout으로 모니터 콘솔에 텍스트 결과 출력하기
        import sys
        data.to_csv(sys.stdout, sep='|')
        
        # 출력에서 손실된값은 빈 문자열로 나타내기
        data.to_csv(sys.stdout, na_rep='NULL')
        
        # 옵션 지정하지 않을 경우 컬럼과 색인 모두 생략하여 표시
        data.to_csv(sys.stdout, index=False, header=False)
        
        # 컬럼의 일부분만 기록하고, 순서 지정
        data.to_csv(sys.stdout, index=False, columns=['a','c', 'b'])
    ```
    
    ![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbQTqFO%2FbtrPQ40ct9J%2FUNPRgs2vCOEvONKdBCnBBk%2Fimg.png)
    

- Series객체를 `to_csv` 메서드를 이용해 파일에 기록하기
    
    ```python
    # Series 객체를 csv로 기록
    def series_to_csv():
        import numpy as np 
        
        # 2000-01-01 부터 2000-01-07까지 날짜 객체 생성
        dates = pd.date_range('1/1/2000', periods=7)
        
        # 0부터 6까지 Series 객체 생성 (index는 dates로 지정)
        ts = pd.Series(np.arange(7), index=dates)
        print(ts)
        
        ts.to_csv('examples/tseries.csv') # 파일에 저장
        
        print(pd.read_csv('examples/tseries.csv'))
    ```
    
    ![Untitled](https://blog.kakaocdn.net/dn/bxMVCg/btrPUmTkGo2/t05prc44IeMeGSvnjRYQMk/img.png)
    


### 구분자 형식 다루기

- `pandas.read_table`  함수를 이용해 디스크에 표 형태로 저장된 대부분의 파일 형식을 불러올 수 있지만, 수동으로 처리해야 하는 경우도 있다.
    - 데이터를 불러오는데 실패하게끔 만드는 잘못된 라인이 포함되어 있는 데이터를 전달 받는 경우도 종종 있다.
- 구분자가 한 글자인 파일은 내장 `csv` 모듈을 이용해 처리할 수 있다.
    
    ```python
    # 구분자 형식 다루기
    def separator():
        import csv # 파이썬 내장 csv 모듈 사용
        
        path = 'examples/ex7.csv'
        f = open(path)
        reader = csv.reader(f) # csv.reader 함수에 파일 객체 넘기기
        for line in reader:
            print(line) # 큰 따옴표가 제거된 튜플 얻기
    ```
    
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FPBFJ4%2FbtrQe2UaIHO%2FKwjhPHYEIsFQS9ICbJLVMk%2Fimg.png">
  
    <img src = "">

- 데이터를 원하는 형식의 출력 파일로 구성하기 (`zip` 함수 이용하여 데이터 컬럼 사전 만들기)
    
    ```python
        # 원하는 형태로 데이터 넣기
        with open(path) as file:
            lines = list(csv.reader(file))
        header, values = lines[0], lines[1:]
        
        # 사전 표기법과 로우를 컬럼으로 전치해주는 zip(*values) 이용해 데이터 컬럼 사전 만들기
        data_dict = {h:v for h,v in zip(header,zip(*values))} # header: 컬럼, values: 데이터
        print(data_dict)
    ```
    
     <img src = "">
    

- `csv.wirter` 메소드를 사용해 구분자를 가진 파일 생성하기
    - `pandas` 객체를 통해 전체 데이터를 다룰 수 있기 때문에, `csv` 모듈 사용보다는 `pdnadas`를 활용한 파일 생성이 훨씬 효율적이다.
    
    ```python
    # csv.writer 메소드를 사용해 구분자를 가진 파일 생성
    def separator_writer():
        import csv # 파이썬 내장 csv 모듈 사용
        
        with open('examples/mydata.csv','w') as f:
    				# 한 줄씩 데이터를 write해야 함
            writer = csv.writer(f, dialect='excel')
            writer.writerow(('one','two','three'))
            writer.writerow(('1','2','3'))
            writer.writerow(('4','5','6'))
            writer.writerow(('7','8','9'))
    ```
    
     <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2Fd0PbBE%2FbtrQeo4MRrY%2FDikqnu5lSsZ8rl847W5cYk%2Fimg.png">
    

### JSON 문자열 다루기

- ********************JSON(Javascript Object Notation)******************** : 웹 브라우저와 다른 애플리케이션이 HTTP 요청으로 데이터를 보낼 때 널리 사용하는 표준 파일 형식
    - CSV 같은 표 형식의 텍스트보다 좀 더 유연한 데이터 형식
    - 기본 자료형으로는 객체(사전), 배열(리스트), 문자열, 숫자, 불리언, Null 을 포함
    - NoSQL 데이터 (대표적으로 MongoDB - mobile side에서 자주 사용되는 DB) 를 다루는데 유용
- 파이썬 표준 라이브러리 `json` 을 사용해 json 데이터를 처리해보자.
    - `json.loads()` : json 문자열을 파이썬 형태로 변환하기
    - `json.dumps()` : 파이썬 객체를 JSON으로 변환
    
    ```python
    # JSON 데이터 다루기
    def json_data():
        # obj에 json 객체 저장하기
        obj = """
        {
            "name":"Wes",
            "places_lived":["US", "Spain", "Germany"],
            "pets":None,
            "siblings":[
                {
                    "name":"Scott",
                    "age":30,
                    "pets":["Zeus", "Zuco"]
                },
                {
                    "name":"Katie",
                    "age":38,
                    "pets":["Stache","Cisco"]                
                }
            ]
         }
         """
         
        # Json 문자열을 파이썬 형태로 변환하기 위한 함수 json.loads
        result = json.loads(obj)
        print(result['name'])
        asjson = json.dumps(result) # 파이썬 객체를 JSON 형태로 변환
        print(asjson)
        
        sibings = pd.DataFrame(result['siblings'], columns=['name','age'])
    ```
    

- `to_json()` : pandas 데이터를 json으로 저장하기 **
    
    ```python
    def json_data_practice():
        data = pd.read_json('examples/example.json') # Json 데이터 읽기
        print(data)
        
        # pandas 데이터를 json으로 저장하기 : to_json() 함수
        print(data.to_json())
        print(data.to_json(orient='records'))
    ```
    
    <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FbxnsJw%2FbtrQbtyYK8i%2FYjuRQLtgNkzrpvPtB1vkjK%2Fimg.png">

    <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2F7Lq69%2FbtrQeNpnHUz%2Fmarf4vskIEUitxsAkEuxKk%2Fimg.png">
    

## 웹 데이터 다루기: HTML, XTML

### HTML 스크래핑

- 파이썬에서는 `lxml`, `BeautifulSoup` , `html5lib` 과 같은 HTML과 XML 형식의 웹 데이터를 읽고 쓸 수 있는 라이브러리가 많다. 그중에서도 `lxml`은 가장 빠르게 동작하고 깨진 HTML과 XML 파일도 잘 처리해줌
- 내장 함수 `read_html()` : lxml이나 BeautifulSoup` 같은 라이브러리를 사용해 자동으로 HTML 파일을 파싱하여 DataFrame으로 변환해준다.
    - <table> 태그에 포함된 모든 표 형식의 데이터에 대한 구문 분석을 수행

### 예제

```python
# html 웹 스크래핑
def html_scrapping():
    tables = pd.read_html('examples/fdic_failed_bank_list.html')
    print(len(tables))
    
    failures = tables[0]
    print(failures.head)
    print(failures.columns)
    print(pd.DataFrame(failures['Closing Date']))
    
    # 데이터 정제, 연도별 부도은행 수 계산 등의 분석

		# Clsoing Date 칼럼의 날짜 데이터 가져오기
    close_timestamps = pd.to_datetime(failures['Closing Date'])
    print(close_timestamps)
    
    cnt = close_timestamps.dt.year.value_counts() # year별로 카운트값 구하기
    print(cnt)
```

- 전체 데이터 확인
    
    <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FEyYEy%2FbtrQeNv8J4e%2FgIkVj76rUQ2kukGnOwA1HK%2Fimg.png">
    
- `Closing Date` 컬럼의 데이터 확인
    
    <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FET8vr%2FbtrQdzrUmeZ%2FAQJqC1BFMCKaX3sGPuONW0%2Fimg.png">
    
- `pd.date_time` 객체로 변환하여 데이터 확인
    
    <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2Feayeaj%2FbtrQaJWji95%2FaCOLaFfH8vBu1Y2iKWXn0k%2Fimg.png">
    

### XML 스크래핑

- **********************************************************XML(Extensible Markup Lanauge**********************************************************) : 계층적 구조와 메타 데이터를 포함하는 중첩된 데이터 구조를 지원하는 데이터 형식
- xml은 구조적으로 html보다 범용적이다. lxml을 이용해 XML 형식의 데이터를 파싱해보자.

### 예제

- 뉴욕 MTA에서 공개하는 버스, 전철 운영 데이터
- xml 파일로 제공되는 실적 자료 xml 파일을 읽어 보자.

```python
# xml 웹 스크래핑
def xml_scrapping():
    from lxml import objectify
    
    path='examples/Performance_MNR.xml' # 전철 실적 데이터
    parsed = objectify.parse(open(path)) # xml 파일 파싱
    root = parsed.getroot() # 루트 노드에 대한 참조
    
    data = []
    skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ', 'DESIRED_CHANGE', 'DECIMAL_PLACES'] # 제외할 컬럼들
    
    for elt in root.INDICATOR:
        el_data = {}
        for child in elt.getchildren():
            if child.tag in skip_fields:
                continue
            el_data[child.tag] = child.pyval # 태그:데이터 사전에 추가
        data.append(el_data)
        
    perf = pd.DataFrame(data)
    print(perf.head())
    
    # xml 데이터 얻기
    from io import StringIO
    tag = '<a href="http://www.google.com">Google</a>'
    root = objectify.parse(StringIO(tag)).getroot() # 루트 노드
    print(root)
    print(root.getchildren())
    print(root.get('href'))
    print(root.text)
```

 <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FMfxJB%2FbtrQeujvd4D%2F0uNZ0qt5imK47kLUofnMtk%2Fimg.png">

<br>

# 2. 이진 데이터 형식

- 데이터를 효율적으로 저장하는 가장 손쉬운 방법은 파이썬에 기본적으로 내장되어 있는 pickle ****************직렬화****************를 사용해 데이터를 이진 형식으로 저장하는 것
    - 오래 보관할 필요가 없는 데이터일 경우에만 추가. 버전 업 된 경우 읽어오지 못할 가능성 존재
- pandas 객체는 pickle을 이용해 데이터를 저장하는 `to_pickle()` 메서드를 사용해 이진 데이터 형식으로 저장
- 이진 데이터 파일을 읽을 때는 `pd.read_pickle()` 메서드를 사용한다

```python
# 이진 데이터 형식
def binary_data():
    frame = pd.read_csv('examples/ex1.csv')
    print(frame)
    
    frame.to_pickle('examples/frame_pickle') # 이진형식으로 데이터 저장 (직렬화된 객체 저장)
    binary_frame = pd.read_pickle('examples/frame_pickle') # read_pickle()로 읽기
    print(binary_frame)
```

- pandas는 HDF5와 Message-Pack 두가지 바이너리 포맷을 지원한다.
- Numpy를 위한 다른 저장 형식도 존재함
    - Bcolz : Blocks 압축 알고리즘에 기반한 압축이 가능한 컬럼지향 바이너리 포맷
    - Feather: 아파치 에로우의 메모리 포맷 사용

### HDF5

- ********HDF5 (Hierarchial Data Format)******** : 대량의 과학 계산용 배열 데이터를 저장하기 위한 계층적 데이터 파일 포맷
- 여러 데이터셋을 저장하고 부가 정보를 기록할 수 있다. 다양한 압축 기술을 사용해 온더플라이(on-the-fly, 실시간) 압축을 지원하며 반복되는 패턴을 가진 데이터를 효과적으로 저장 가능
- 메모리에 모두 적재할 수 없는 엄청나게 큰 데이터를 아주 큰 배열에서 필요한 작은 부분들만 효과적으로 읽고 쓰는데 유용
- `HDFStore` 클래스는 사전처럼 작동하므로, dataframe 객체처럼 다룰 수 있다
- `fixed` 와 `rable` 두 가지 저장 스키마를 지원
    
    ```python
    # HDF5 형식의 이진 데이터
    def HDF5_format():
        frame = pd.DataFrame({'a': np.random.randn(100)})
        store = pd.HDFStore('mydat.h5') # 바이너리 파일 객체 저장 
        store['obj1'] = frame # dataframe 객체를 저장
        store['obj1_col'] = frame['a']
        
        print(store)
        print(store['obj1'], store['obj1_col'])
        
        
        # obj2에 table포맷으로 frame객체 저장
        store.put('obj2', frame, format='table')
        print(store.select('obj2',where=['index>=10 and index<=15'])) # 쿼리 연산 지원
        
        store.close()
        
        # frame 객체를 바로 hdf로 저장
        frame.to_hdf('mydata.h5', 'obj3', format='table')
        print(pd.read_hdf('mydata.h5', 'obj3', where=['index<5']))
    ```
    
    <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FdvvG57%2FbtrQb22bsef%2F4kAgZPgIhHK0nqvQvquKlK%2Fimg.png"> <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FkJ9z7%2FbtrQepvOUmn%2FiC4Uki4pGYOpvk1PSbUK4K%2Fimg.png">
    

### MS 엑셀 파일 다루기

- `ExcelFile` 클래스나 `pandas.read_excel()` 함수를 사용해 엑셀 데이터를 읽어낼 수 있도록 지원함
- xls, xlsx 파일을 읽기 위해 xlrd, openpyxl 패키지를 사용
    
    ```python
    # 엑셀 파일 다루기
    def excel_data():
        xlsx = pd.ExcelFile('examples/ex1.xlsx')
        print(pd.read_excel(xlsx, 'Sheet1')) # xlsx 파일의 시트 읽기
        
        frame = pd.read_excel('examples/ex1.xlsx', 'Sheet1')
        
        # pd 데이터를 엑셀 파일로 저장하기
        writer = pd.ExcelWriter('examples/ex2.xlsx')
        frame.to_excel(writer, 'Sheet1')
        # frame.to_excel('examples/ex2.xlsx')
        writer.save()
    ```
    

<br>


# 3. 웹 API와 함께 사용하기

- 데이터 피드를 JSON 이나 다른 형식으로 얻을 수 있게 공개 API를 제공하는 웹사이트가 많다.
- 많은 패키지 중, `requests` 패키지를 이용해 파이썬으로 API를 사용해보자
- `GET` http 요청을 생성하여 pandas 깃허브에서 최근 30개의 이슈를 가져와보자.
    
    ```python
    # requests 이용해 http api 이용하기
    def web_api_http():
        import requests
        
        url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
        resp = requests.get(url) # GET 요청을 보내보자
        
        print(resp) # 응답 http status 코드: 200(성공) or else
        
        # 깃허브 이슈 페이지(댓글 제외)에서 찾을 수 있는 모든 데이터 추출
        data = resp.json() # 응답 json 데이터를 파이썬 사전 형태로 변환
        print(data[0]['title'])
        
        # DataFrame객체로 생성하고 관심 있는 필드만 추출하기
        issues = pd.DataFrame(data, columns=['number', 'title',' labels', 'state'])
        print(issues)
    ```

    <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FAzBMu%2FbtrQfDmiZWu%2F3moMQLUrhoqn0PCHxcJAWk%2Fimg.png">

<br>


# 4. 데이터베이스와 함께 사용하기

- SQL 기반의 관계형 DB(SQL, PostgreSQL, MySQL) 에서 데이터를 읽어 와 DataFrame에 저장해보자
- 파이썬 내장 `sqlite3` 드라이버를 사용하여 SQLite db를 이용해보자.
- `SQLAlchemy` 를 이용한 연결을 통해 쉽게 db를 다룰 수 있음