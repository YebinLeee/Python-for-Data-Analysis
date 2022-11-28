import pandas as pd 
import numpy as np
from numpy import nan as NA
import re 

def find_null_data():
    # 누락값 np.nan은 NaN으로 표시 
    string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
    print(string_data)
    print(string_data.isnull())
    
    # 파이썬의 내장 None 값 또한 NaN으로 인식
    string_data[0] = None
    print(string_data.isnull())

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
    cleaned = data.dropna() # 하나라도 NA를 포함하고 있는 축은 제외
    print(cleaned)
    
    # 모두 NA 값인 로우만 제외
    cleaned_all_none = data.dropna(how='all') 
    
    data[4] = NA
    print(data)
    
    # 전체가 NA인 컬럼 제외하기
    cleaned_column = data.dropna(axis=1, how='all')
    print(cleaned_column)


# 값이 특정 개수 이상 들어있는 로우만 선택  
def dropna_threshold():
    df = pd.DataFrame(np.random.randn(7,3))
    df.iloc[:4,1]=NA
    df.iloc[:2,2]=NA
    print(df)
    print(df.dropna())
  
    # thresh를 2로 설정하여 2개 이상의 값이 들어있는 로우에 대해서만 DF 객체 전달
    print(df.dropna(thresh=2))
    
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
   
# 데이터 형태 변환하기
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
    
# 값 치환하기
def data_replacement():
    data = pd.Series([1., -999., 2., -999., -1000., 3.])
    print(data)
    
    # 누락된 값을 나타내는 -999를 pandas에서 인식할 수 있는 NA 값으로 치환하기
    data.replace(-999, np.nan)
    data.replace([-999, -1000], np.nan) # 한 번에 여러 값 치환하기
    data.replace([-999, -1000], [np.nan, 0]) # 치환하는 값마다 다른 값으로 치환하기
    data.replace({-999: np.nan, -1000:0}) # 사전을 이용하기 가능
    
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

# 그룹 분석    
def discretize():
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

# 치환과 임의 샘플링
def permutatation_sampling_replacement():
    df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
    print(df)
    
    sampler = np.random.permutation(5) # 0부터 4까지 숫자를 임의로 재배치
    print(sampler)
    
    # sampler에 의해 take() 메서드로 배치 변경
    print(df.take(sampler))
    
    # sample()로 치환 없이 일부만 임의로 선택 
    print(df.sample(n=3)) 
    
    # 반복 선택을 허용하여 표본을 치환을 통해 생성해 내기 (replace=True 옵션 지정)
    choices = pd.Series([5,7,-1,6,4])
    draws = choices.sample(n=10, replace=True) # choices 배열의 값으로 반복 허용하여 10개의 샘플 생성
    print(draws)
    
# 표시자/더미 변수 계산하기
def dummies():
    df = pd.DataFrame({'key':['b','b','a','c','a','b'],
                       'data1':range(6)})
    print(pd.get_dummies(df['key'])) # key의 값이 컬럼이 되고, 해당되는 값에 1, 아닌 경우 0을 지정
    
    # 컬럼에 접두어(prefix) 추가하고 다른 데이터와 병합하기
    dummies = pd.get_dummies(df['key'], prefix='key') # 컬럼이 key_a, key_b, key_c로 변경
    df_with_dummy = df[['data1']].join(dummies) # data1과 dummies 병합
    print(df_with_dummy)
    
 
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
    print(len(genres), genres)
    
    # 표시자 DataFrame 생성
    zero_matrix = np.zeros((len(movies), len(genres)))
    dummies = pd.DataFrame(zero_matrix, columns=genres)
    print(dummies)
    
    # 각 영화 순회하며 dummies의 각 로우 항목을 1로 설정
    gen = movies.genres[0]
    print(gen.split('|'))
    dummies.columns.get_indexer(gen.split('|'))
    
    # 전체 영화 데이터에 genre 인덱서 지정
    for i, gen in enumerate(movies.genres):
        indices = dummies.columns.get_indexer(gen.split('|'))
        dummies.iloc[i, indices] = 1
         
    # movies와 조합하기
    movies_windic = movies.join(dummies.add_prefix('Genre_')) # Prefix 지정
    print(movies_windic.iloc[0]) # 첫번째 데이터 정보 얻어오기 (Toy Story)

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
    
# 정규 표현식 활용하기
def regular_expression():
    #하나 이상의 공백 문자를 의미하는 \s+를 사용하여 문자열 분리
    text = "foo bar\t baz \tqux"
    print(re.split('\s+', text)) # 정규 표현식 컴파일 후 split 메서드 실행
    
    # 직접 정규 표현식을 컴파일하여 얻은 정규 표현식 객체를 재사용하기
    regex = re.compile('\s+')
    print(regex.split(text))
    
    # 정규 표현식에 매칭되는 모든 패턴의 목록 얻기
    print(regex.findall(text))
    
# 이메일 예제로 정규 표현식 이해하기
def regular_expression_email():  
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
    
def main():
    # find_null_data()
    # dropna_method()
    # dropna_threshold()
    # fillna_method()
    # duplicated_method()
    # data_transformation_by_mapping()
    # data_replacement()
    # change_column_name()
    # discretize()
    # handle_outlier()
    # permutatation_sampling_replacement()
    # dummies()
    # movies_with_categories()
    # string_method()
    # regular_expression()
    # regular_expression_email()
    vectorized_string_method()
    
if __name__ == '__main__': # main()
    main()