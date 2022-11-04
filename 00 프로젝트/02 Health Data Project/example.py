import pandas as pd 
import numpy as np
import csv 
import json

def data_info():
    path = '00 프로젝트/02 Health Data Project/data/data.csv' # 20221104_238GRC.csv
    
    data = pd.read_csv(path)
    print(data.head())
    print(data.describe())
    
    frame = pd.DataFrame(data)
    print(frame)
    
    json_data = frame.to_json()
    print(json_data)
    tojson = json.loads(json_data)
    asjson = json.dumps(json_data)
    
    
    
''' 
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
'''

def main():
    data_info()
    
        
if __name__ == '__main__': # main()
    main()