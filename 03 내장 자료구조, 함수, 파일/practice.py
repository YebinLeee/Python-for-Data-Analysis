
def tupleInfo():
    tup = 1,2,3
    print(tup)

    # 중첩된 튜플
    nested_tup = (4,5,6),(7,8)
    print(nested_tup)

    print(tuple([5,0,3])) # 순차 데이터를 tuple()을 이용해 튜플로 변경

    tup = tuple("sample")
    print(tup) # ('s', 'a,' ,,)
    print(tup[0]) # 각 요소는 [인덱스]를 이용해 접근
    
    # 튜플의 객체를 수정 가능하지만, 한번 생성된 슬롯의 객체는 변경 불가
    tup = tuple(["string", [1,2], True])
    # tup[2] = False # 불가
    
    # 튜플의 리스트 값에 요소 추가 가능
    tup = tuple(["foo", [1,2], True])
    tup[1].append(3)
    print(tup)
    
    # 서로 다른 변수로 값 분리 가능
    tup = (4,5,6)
    a,b,c = tup # 대입 연산자를 이용해 여러 변수에 튜플 객체를 할당하며 각 변수에 튜플의 원소값이 할당됨
    print('b : ', b)
    
    a = (1,2,2,2,3,4,2)
    print(a.count(2)) # count() 메소드로 동일 원소의 개수 셀 수 있음
    
import bisect

def listInfo():
    a_list = [2,3,7,None]
    tup = ('foo', 'bar', 'baz')
    b_list = list(tup) # 튜플을 리스트로 변경 가능
    print(b_list) 
    
    b_list[1] = 'test' # 리스트의 요소 변경
    print(b_list)
    
    # 리스트에 요소 추가하기 
    b_list.append('dwarf')
    print(b_list) # 마지막 위치에 새로운 요소 추가
    b_list.insert(1, 'red') # 1번 인덱스 위치에 새로운 요소 추가
    print(b_list)
    
    # 리스트의 요소 삭제하기
    print(b_list.pop(2)) # 2번 인덱스의 요소 삭제
    print(b_list)
    b_list.remove('foo') # 첫번째 요소부터 리스트에서 제거됨
    print(b_list)
    
    # 객체 내부의 요소 확인하기
    print('dwarf' in b_list)
    print('dwarf' not in b_list)
    
    # 정렬
    a = [7,2,5,1,3]
    a.sort() # 오름차순 정렬
    print(a) 
    b = ['saw', 'small', 'He', 'foxes', 'six']
    b.sort(key=len) # 요소들의 길이에 대한 정렬 수행
    print(b)
    
    # 이진탐색 (bisect 모듈)
    c = [1,2,2,2,3,4,7]
    print(bisect.bisect(c,2)) # 새로운 요소가 추가될 때 정렬된 리스트를 유지하기 위한 위치를 리턴
    print(bisect.bisect(c,5))
    bisect.insort(c,6) # 정렬된 리스트를 유지하며 새로운 요소를 리스트 객체에 추가
    print(c)
    
    # 슬라이싱
    seq = [7,2,3,7,5,6,0,1]
    print(seq[1:5]) # 1번~5번 인덱스 요소 출력
    seq[3:4] = [6,3] # 슬라이싱 영역에 데이터 할당
    print(seq)
    # 슬라이싱 시 시작/종료 지점 인덱스 생략 가능
    print(seq[:5])
    print(seq[3:])
    # 슬라이싱 위치에 대한 마이너스 인덱스는 요소의 끝에서부터의 위치
    print(seq[-4:])
    print(seq[:-6:-2])
    
    # 중첩된 콜론 -> 인덱스 스텝 크기결정
    print(seq[::2]) # 처음부터 끝까지 2 간격
    print(seq[::-1]) # 리스트 요소를 역순으로 리턴
    
    
def embeddingFuncInfo():
    # enumerate() : 순차 자료에 대해 개별 요소와 인덱스 함께 생성
    some_list = ['foo', 'bar', 'baz']
    mapping = {} # dic
    for i, v in enumerate(some_list):
        mapping[v]=i
    print(mapping)
    
    # sorted() : 순차 자료에 대한 정렬된 리스트 리턴
    sorted([5,2,1,4,0,3,2])
    
    # reversed() : 역순으로 순회
    print(list(reversed(range(10))))
    
    # zip() : 서로 다른 순차 자료 쌍에 대해 튜플 리스트 생성
    seq1 = ['foo','bar','baz']
    seq2 = ['one', 'two', 'three']
    zipped = zip(seq1, seq2)
    print(list(zipped))
    

def dictionaryInfo():
    empty_dict = {} # Dictionary 초기화
    d1 = {'a':'some_value',
          'b':[1,2,3,4],
          7:'an integer',
          5:'some value'
          }
    print(d1)
    print(d1[7], d1['b'])
    
    # 데이터 요소 삭제
    del d1[5] # 
    print(d1)
    ret= d1.pop('b') # d1['b'] 에 해당하는 key-value 제거
    print(ret)
    print(d1)    
    
    # update() : 두 사전 자료를 연결
    d1.update({'z':'foo', 'q':5})
    print(d1)
    
    # 두 순차 자료를 사전으로 만들기 위해 zip() 사용
    mapping = {}
    key_list = {1,2,3}
    value_list = {10,20,30}
    for key, value in zip(key_list, value_list):
        mapping[key]=value
    print(mapping)
    
    key = 5
    default_value = 'default value'
    some_dict = d1
    if key in some_dict:
        value = some_dict[key]
    else:
        value = default_value
    value = some_dict.get(key, default_value) # 키에 대응되는 값 리턴

def setInfo():
    a = {1,2,3,4,5}
    b = {3,4,5,6,7,8}
    print(a.union(b))  # 합집합 a|b
    print(a.intersection((b))) # 교집합 a&b
    print(a.difference(b)) # 차집합 a-b
    
    a_set = {1,2,3,4,5}
    b_set = {1,2,3}
    print(b_set.issubset(a_set)) # 부분집합
    print(a_set.issubset((b_set))) 
    
def my_function(x,y,z=1.5):
    if z>1:
        return z*(x+y)
    else:
        return z/(x+y)
    
# print(my_function(4,6,3.5))
# print(my_function(10,20))

def func(): 
    a = [] # 지역 변수 - func() 내부에서만 사용 가능
    for j in range(5):
        a.append(j)
    print(a)

# 여러 값 반환
def f():
    a=5
    b=6
    c=7
    return a,b,c
a,b,c = f()
# print(a,b,c)

      
import re
states = ['Alabama', 'Gerogia!', 'Georgia', 'georgia?', 'Fl#oarida', 'florida']

def clean_strings(string):
    result = []
    for value in string:
        value = value.strip()
        value = re.sub('[!#?]' , '', value)  
        value = value.title()
        result.append(value)
    return result 
# print(clean_strings(states))

# 람다 함수
z=2
ret1 = (lambda x:x*2)(z)
# print(ret1)

# 제네레이터
gen = (x**2 for x in range(10))
l = list(gen)
# print(l)

# 예외 처리
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x
# print(attempt_float('1.23'))
# print(attempt_float('something'))

# 파일 IO
path = 'address.txt'
f = open(path)
print(path)
f.close()

# tupleInfo()
# listInfo()
# embeddingFuncInfo()
# dictionaryInfo()
# setInfo()