import pandas as pd 
import numpy as np
from numpy import nan as NA 

def getDataFrame():
    df = pd.DataFrame(np.random.randn(6,3))
    df.iloc[2:,1] = NA
    df.iloc[4:,2] = NA 
    return df 

def nan():
    # np.nan 으로 누락 데이터 넣기
    string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
    print(string_data)
    print(string_data.isnull())
    
    # na값으로 집어넣기
    string_data[0]=None
    print(string_data.isnull())

def filtering_missing_data():
    # dropna 메서드 사용
    data = pd.Series([1, NA, 3.5, NA, 7])
    print(data)
    
    obj = data.dropna() # null이 아닌 데이터와 색인값만 들어있는 Series 반환
    print(obj) 
    print(data[data.notnull()])
    
    # DataFrame 객체
    data2 = pd.DataFrame([[1,6.5,3.],
                         [1,NA,NA],
                         [NA,NA,NA],
                         [NA,6.5,3]])
    print(data)

    cleaned = data2.dropna()
    print(cleaned)
    
    # 모두 NA값인 로우만 제외시키기
    print(data2.dropna(how='all'))
    
    
# 누락된 데이터 처리하기 
def threshhold():
    df = getDataFrame();
    print(df)
    
    obj = df.dropna()
    print(obj)
    
    # 특정 수의 관측값을 퐇마하는 로우를 유지하기 위해 thresh인수 사용
    obj = df.dropna(thresh=2)    

def fillna():
    df = getDataFrame();
    print(df)
    
    # 누락된 데이터를 0으로 대체
    obj = df.fillna(0)
    print(obj)
    
    # 특정 값을 다른 값으로 채우기
    obj = df.fillna({1:0.5, 2:0})
    print(obj)
    
    # 새로운 객체 반환 대신 기존 객체를 변경
    df.fillna(0, inplace=True)
    print(df)
    
    # 재인덱싱에 사용할 수 있는 동일한 보간 방법 사용
    df = getDataFrame();
    obj = df.fillna(method='ffill')
    print(obj)
    
    obj = df.fillna(method='ffill', limit=2)
    print(obj)
    
# 중복 제거하기]
def removeDuplicates():
    data = pd.DataFrame({'k1':['one','two']*3 + ['two'],
                         'k2':[1,1,2,3,3,4,4]})
    print(data)
    
    obj = data.duplicated()
    print(obj)
    
    # drop_duplicates()로 중복된 배열이 false인 DataFrame객체를 리턴
    obj = data.drop_duplicates()
    print(obj)
    
    obj = data.drop_duplicates(['k1'])
    
    obj = data.drop_duplicates(['k1','k2'])
    print(obj)
    
    # 마지막 발견된 데이터를 유지
    obj = data.drop_duplicates(['k1','k2'], keep='last')
    

# 데이터 수정하기 
def replaceValues():
    data = pd.Series([1., -999., -1000., 3.])
    obj = data.replace(-999, np.nan) # 오류 데이터를 NA 값으로 변경
    print(obj)
    
    # 한번에 여러 데이터 변경
    obj = data.replace([-999,-1000], np.nan)
    print(obj)
    
    # 각 데이터에 대해 다른 변경 데이터를 사용하기 위해서
    obj = data.replace([-999,-1000], [np.nan, 0])
    print(obj) 
    
    # 사전으로 replace
    obj = data.replace({-999:np.nan, -1000:0})
    print(obj)
    
    
# 데이터 변형하기

# 축 이름 변경하기
def renameAxisIndexes():
    data = pd.DataFrame(np.arange(12).reshape((3,4)),
                        index = ['Ohio', 'Colorado', 'New Yrok'],
                        columns = ['one', 'two', 'three', 'four'])
    # 앞에 4번 인덱스까지 자르고 대문자로 변경
    transform = lambda x: x[:4].upper()
    obj = data.index.map(transform)
    print(obj)
    
    # 제자리에서 인덱스 수정
    data.index = data.index.map(transform)
    print(data)
    
    # 원본 데이터 수정 없이 데이터 셋의 변형된 버전 생성
    obj = data.rename(index=str.title, columns=str.upper)
    print(obj)
    
    # 사전 객체를 통해 축 레이블의 하위집합 이름 변경
    obj = data.rename(index={'OHIO':'INDIANA'}, columns={'three':'peekaboo'})
    print(obj)
    
    # 데이터 셋을 제자리에서 수정
    data.rename(index={'OHIO':'INDIANA'}, inplace=True)
    print(data)
    
# 이산화 및 양자화 (연속 데이터를 이산화하거나 분석을 위해 양자로 분리)
def discretAndBinning():
    ages = [20,22,25,27,21,23,37,31,61,45,41,32]
    bins = [18,25,35,60,100]
    
    # ages 단위로 그룹 구성 (18~25, 26~35, 36~60, 61~100)
    cats = pd.cut(ages, bins)
    print(cats)
    
    print(cats.codes) 
    print(cats.categories) # 카테고리 이름
    
    # cut 결과에 대한 그룹 수 
    obj = pd.value_counts(cats)
    
    # 레이블 옵션에 배열을 전달하여 고유한 양자 이름 전달
    group_names = ['Youth','YoungAdult','MiddleAged','Senior']
    obj = pd.cut(ages, bins, labels=group_names)
    print(obj)
    
    # 명시적인 bin 경계 대신 잘라낼 정수 개수의 bin 전달하여 데이터의 최소값, 최대값을 기반으로 동일한 길이의 bin 계산
    data = np.random.rand(20)
    obj = pd.cut(data,4,precision=2)
    print(obj)
    
def discretByQcut():
    # 샘플 수를 기반으로 데이터를 나누는 qcut
    data = np.random.randn(1000)
    cats = pd.qcut(data, 4) # 샘플 분위수를 사용하여 정의에 따라 동일한 크기의 빈으로 나눔
    print(cats) 
    
    obj = pd.value_counts(cats)
    print(obj)
    
    
# 특잇값 검출 및 제외
def outlier():
    data = pd.DataFrame(np.random.randn(1000,4))
    print(data.describe())
    
    # 절댓값이 3을 초과하는 값 찾아내기
    col = data[2]
    print(col[np.abs(col) > 3])
    
    # 3 초과하는 값이 들어있는 모든 로우 선택
    print(data[np.abs(data) > 3].any(1))
    
    # 절댓값이 3을 초과하는 값을 -3, 3으로 변경
    data[np.abs(data) > 3] = np.sign(data) * 3
    print(data.describe())
    
    # 양수, 음수에 따라 1, -1 베열 반환
    print(np.sign(data).head())
    
# 치환과 읨의 샘플링
def permutate():
    df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
    sampler = np.random.permutation(5) # 임의ㅅ의 순서로 재배치
    print(sampler)
    
    print(df)
    print(df.take(sampler))
    
    # 치환 없이 일부만 임의로 선택
    print(df.sample(n=3))
    
    # 반복 선택을 허용하여 표본을 치환으로 생성
    choices = pd.Series([5,7,-1,6,4])
    draws = choices.sample(n=10, replace=True)
    print(draws)
    
def movie_dummies_example():
    df = pd.DataFrame({'key':['b','b','a','c','a','b'],
                       'data1':range(6)})
    print(pd.get_dummies(df['key']))
    
    # 다른 데이터와 병합
    dummies = pd.get_dummies(df['key'], prefix='key')
    df_with_dummy = df[['data1']].join(dummies)
    print(df_with_dummy)
    
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table('datasets/movielens/movies.dat',sep='::', header=None, names=mnames, engine='python', encoding='ISO-8859-1')
    
    # 각 장르마다 표시자값 추가
    all_genres = []
    for x in movies.genres:
        all_genres.extend(x.split('|'))
        genres = pd.unique(all_genres)
    print(genres)
        
    zero_matrix = np.zeros((len(movies), len(genres)))
    dummies = pd.DataFrame(zero_matrix, columns=genres)
    
    gen = movies.genres[0]
    print(gen.split('|'))
    
    dummies.columns.get_indexer(gen.split('|'))
    # 색인에 맞게 값 대입
    for i,gen in enumerate(movies.genres):
        indices = dummies.columns.get_indexer(gen.split('|'))
        dummies.iloc[i, indices] = 1
    
    movies_windic = movies.join(dummies.add_prefix('Genre_'))
    print(movies_windic.iloc[0])
    
    # 통계 이용
    np.random.seed(12345)
    values = np.random.rand(10)
    print(values)
    
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    print(pd.get_dummies(pd.cut(values, bins)))
    
# 문자열 다루기
def string_list():
    # 문자열 분리
    val = 'a,b, guido'
    print(val.split(','))
    
    # 공백문자 제거
    pieces = [x.strip() for x in val.split(',')]
    print(pieces)
    
    # 분리된 문자열 더하기
    first, second, third = pieces
    print(first + '::' + second + '::' + third)
    print('::'.join(pieces))
    
    # 부분 문자열의 위치 찾기
    print('guido' in val)
    print(val.index(','))
    print(val.find(':'))
    
    # index의 경우 발견 못하면 예외 발생, find는 -1 반환
    # print(val.index(':'))
    
    # 문자열 발견된 횟수 세기
    print(val.count(','))
    
    # 찾아낸 패턴을 다른 문자열로 치환
    print(val.replace(',', '::'))
    print(val.replace(',',''))
    
# 정규 표현식
def regular_expression():
    import re
    
    # 공백 문자 포함된 문자열 나누기
    text = "foo bar\t baz \tqux"
    print(re.split('\s+', text))
    
    # compile 된 후에 split 메서드 실행
    regex = re.compile('\s+')
    print(regex.split(text))
    print(regex.findall(text)) # 매칭되는 모든 패턴의 목록 찾기
    
    text="""Dave dave@google.com
    Steve steve@gmail.com
    Rob rob@gmail.com
    Ryan ryan@yahoo.com
    """
    
    pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
    regex = re.compile(pattern, flags=re.IGNORECASE) # 대소문자 구분 X
    
    # 이메일 주소 리스트 생성하기
    print(regex.findall(text))
    
    m = regex.search(text)
    print(m)
    print(text[m.start():m.end()])
    
    print(regex.match(text))
    print(regex.sub('READACTED', text))
    
    # 이메일 주소를 찾아 동시에 각 이메일 주소를 사용자, 도메인 이름, 도메인 접미사 세가지 컴포넌트로 나누는 경우  각 패턴을 괄호로 묶어줌
    pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
    regex = re.compile(pattern, flags=re.IGNORECASE)
    
    m = regex.match('wesm@bright.net')
    print(m.groups())
    
    print(regex.findall(text))
    print(regex.sub(r'Username: \1, Domain: \2, suffix: \3', text))
    
# 벡터화된 문자열 함수
def vectorized_string():
    import re
    data = {'Dave':'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob':'rob@gmail.com', 'Wes':np.nan}
    data = pd.Series(data)
    print(data)
    
    print(data.isnull())
    
    # 포함하는 문자열 확인
    print(data.str.contains('gmail'))
    pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
    print(data.str.findall(pattern, flags=re.IGNORECASE))
    
    matches = data.str.match(pattern, flags=re.IGNORECASE)
    print(matches)
    # print(matches.str.get(1))
    # print(matches.str[0])
    print(data.str[:5])
    
    
def main():
    # nan()
    # filtering_missing_data()
    # threshhold()
    # fillna()
    # removeDuplicates()
    # replaceValues()
    # renameAxisIndexes()
    # discretAndBinning()
    # discretByQcut()
    # outlier()
    # permutate()
    # movie_dummies_example()
    # string_list()
    # regular_expression()
    vectorized_string()
    
main()