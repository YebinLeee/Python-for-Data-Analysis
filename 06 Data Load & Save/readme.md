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

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FAvezV%2FbtrPS0C1wc9%2F69Y9d4poc7KL4z4kKpuUY0%2Fimg.png)

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcJwaq4%2FbtrPVDtKTlg%2FD7wv38xg76lW5GN2gKozU1%2Fimg.png)

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
    