import pandas as pd 
import numpy as np
import csv 
import json

# 구분자 형식 다루기
def separator():
    path = 'examples/ex7.csv'
    f = open(path)
    reader = csv.reader(f) # csv.reader 함수에 파일 객체 넘기기
    for line in reader:
        print(line) # 큰 따옴표가 제거된 튜플 얻기
    
    # 원하는 형태로 데이터 넣기
    with open(path) as file:
        lines = list(csv.reader(file))
    header, values = lines[0], lines[1:]
    
    # 사전 표기법과 로우를 컬럼으로 전치해주는 zip(*values) 이용해 데이터 컬럼 사전 만들기
    data_dict = {h:v for h,v in zip(header,zip(*values))} # header: 컬럼, values: 데이터
    print(data_dict)
    
# csv.writer 메소드를 사용해 구분자를 가진 파일 생성
def separator_writer():   
    with open('examples/mydata.csv','w') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(('one','two','three'))
        writer.writerow(('1','2','3'))
        writer.writerow(('4','5','6'))
        writer.writerow(('7','8','9'))
        
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
    
    
def json_data_practice():
    data = pd.read_json('examples/example.json') # Json 데이터 읽기
    print(data)
    
    # pandas 데이터를 json으로 저장하기 : to_json() 함수
    print(data.to_json())
    print(data.to_json(orient='records'))
  
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
    
# 이진 데이터 형식
def binary_data():
    frame = pd.read_csv('examples/ex1.csv')
    print(frame)
    
    frame.to_pickle('examples/frame_pickle') # 이진형식으로 데이터 저장 (직렬화된 객체 저장)
    binary_frame = pd.read_pickle('examples/frame_pickle') # read_pickle()로 읽기
    print(binary_frame)
    
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
    
    
def database_data():
    import sqlite3
    
    # 테이블 생성
    query = """
        CREATE TABLE test (a varchar(20), b varchar(20), c REAL, d INTEGER);
    """
    
    # sqlite3 연결 후 쿼리 전송, 커밋
    con = sqlite3.connect('mydata.sqlite')
    con.execute(query)
    con.commit()

    # 데이터 values 입력
    data = [('Atalanta', 'Gerrgia', 1.25, 6),
            ('Tallahassee', 'Florida', 2.6, 3),
            ('Sacramento', 'California', 1.7, 5)]
    
    stmt = "INSERT INTO test VALUES(?,?,?,?)"
    con.executemany(stmt, data)
    con.commit()
    
    cursor = con.execute('select * from test')
    rows = cursor.fetchall()
    print(rows)
    
    print(cursor.description)
    frame = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
    print(frame)
    
    # SQLAlchemy 사용하여 db 사용
    import sqlalchemy as aqla
    
    db = sqla.create_engine('sqlite:///mydata.sqlte')
    print(pd.read_sql('select * from test', db))
    
    
    
def main():
    # separator()
    # separator_writer()
    # json_data_practice()
    # html_scrapping()
    # xml_scrapping()
    # binary_data()
    # HDF5_format()
    # excel_data()
    # web_api_http()
    database_data()
    
if __name__ == '__main__': # main()
    main()