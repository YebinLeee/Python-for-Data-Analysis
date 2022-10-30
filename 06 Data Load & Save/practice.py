import pandas as pd

# CSV 파일들 열어 출력해보기
def read_csv_file():
    for i in range(1,8):
        path = 'examples/ex{0}.csv'.format(i)
        df = pd.read_csv(path)
        df = pd.read_table(path, sep=',') # read_table의 구분자를 쉼표로 지정
        
        # 컬럼 이름 자동 생성 (0~)
        df = pd.read_csv('examples/ex2.csv', header=None)
        
        # 직접 컬럼 이름 지정하기
        df = pd.read_csv('examples/ex2.csv', names=['a','b','c','d','message'])
        
        # message 컬럼을 색인으로 지정하기
        names=['a','b','c','d','message']
        df = pd.read_csv('examples/ex2.csv', names=names, index_col='message')
        
        print(df, end='\n\n\n') # 자동으로 index값 부여 (column이름은 header)

# 계층적 인덱스 설정
def csv_index():
    print(pd.read_csv('examples/csv_mindex.csv'))
    # 다중 컬럼으로부터 계층적 인덱스 설정하기
    parsed = pd.read_csv('examples/csv_mindex.csv', index_col=['key1','key2'])
    print(parsed)
    
# read_table을 이용해 인덱스 추론, 공백 문자를 기준으로 데이터값들 자르기
def re_seperator():
    print(list(open('examples/ex3.txt')))
    result = pd.read_table('examples/ex3.txt',sep='\s+')
    print(result)

# 로우 제외하기
def skip_rows():
    orig = pd.read_csv('examples/ex4.csv')
    print(orig)
    ret = pd.read_csv('examples/ex4.csv', skiprows=[0,2,3]) # 0,2,3번 로우 건너뛰기
    print(ret) 
    
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
    
    
def read_parts():
    pd.options.display.max_rows=10 # 최대 10개의 데이터만 출력
    result = pd.read_csv('examples/ex6.csv')
    print(result, end='\n\n')
    
    # nrows 옵션 설정하기
    print(pd.read_csv('examples/ex6.csv', nrows=5))
    
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
    
    
     
    
def main():
    # read_csv_file()
    # csv_index()
    # re_seperator()
    # skip_rows()
    # missing_data()
    # read_parts()
    # read_chunk()
    # to_csv_export()
    # series_to_csv()
    
if __name__ == '__main__': # main()
    main()