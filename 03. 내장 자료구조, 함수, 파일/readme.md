# CH3. 내장 자료구조, 함수, 파일

# 자료구조와 순차 자료형

## 1. 튜플

- 1차원의 고정된 크기를 가지는 변경 불가능한 순차 자료형
    
    ```python
    tup = tuple(['foo', [1,2,], True])
    tup[1].append(3)
    
    nested_tup = (1,2,3), (4,5,6)
    
    print(tuple('string')) # ('s', 't', 'r', 'i', 'n', 'g')
    print(tup[0]) # 인덱스로 접근 가능
    
    ```
    
- 튜플에서 값 분리하기 (대입 연산자 사용)
    
    ```python
    tup = (4,5,6)
    a,b,c, = tup # a=4, b=5, c=6
    
    # 쉽게 두 값의 교환 가능
    a,b = 1,2
    a,b = b,a
    ```
    
- 튜플 순회
    
    ```python
    seq = [(1,2,3), (4,5,6), (7,8,9)]
    for a,b,c, in seq:
    		print(a,b,c)
    
    # 포인터 개념을 이용해 긴 인자를 담기
    values = 1,2,3,4,5,6
    a,b,*rest = values # a=1, b=2, *rest=3,4,5,6
    
    # 불필요한 변수라는 것을 나타내기 위해 _를 사용
    c, d, *_ = values
    ```
    
- 튜플 메서드 : `count` 함수
    
    ```python
    tup = (1,2,2,2,2,3,3,4)
    print(tup.count(2)) # 4 출력
    ```
    
<br>


## 2. 리스트

- 객체의 1차원 순차 자료형이며, 튜플과 다르게 크기나 내용의 변경이 가능
- 리스트 생성
    
    ```python
    a_list = [1,2,3,None]
    tup = (1,2,3,4)
    
    # 튜플을 리스트로 변경하기
    tup_list = list(tup)
    tup_list[1] = 3 # 인덱스로 접근 후 값 변경
    
    # 이터레이터에서 실제 값을 모두 담기 위한 용도로 사용
    gen = range(5,15)
    print(list(gen))
    ```
    
- 원소 추가 및 삭제 : `append()` , `insert(n,value)` , `remove(value)` , `pop(n)`
    
    ```python
    b_list = ["hello", "world", "hi"]
    b_list.append("dwarf")
    b_list.insert(3, "hello") # insert는 연산 비용이 많이 드므로 collections.deque 사용을 권장
    
    b_list.pop(2)
    b_list.remove("hello") # 제일 앞에 위치한 첫번째 값 삭제
    
    print("foo" in b_list) # False
    ```
    
- **리스트 이어붙이기** : `+` 연산자, `extend([arr])`
    - 리스트를 이어붙이는 경우 새로운 리스트를 생성하고 값을 복사하므로 상대적으로 연산 비용이 높다.
    - 큰 리스트일수록 extend를 사용해서 기존릐 리스트에 값을 확장하여 추가하는 것이 더 적절하다.
    
    ```python
    x = [4, None, "foo"] + [5, "hi"]
    x.extend([5, "hello", (2,3)]) # extend 메소드 사용
    ```
    
- **정렬**
    - `sort()`
    - `sorted()` 함수 : 정렬하여 복사본 생성
    
    ```python
    a = [7,2,5,3,1]
    a.sort() # a = [1,2,3,5,7]
    
    b = ["boy", "absolute", "cat"]
    b.sort(key=len) # 길이 순 정렬
    ```
    
- **이진 탐색과 정렬된 리스트 유지하기**
    - `bisect` 모듈 : 정렬된 리스트에 대한 이진 탐색과 정렬을 유지하며 값을 추가하는 기능 제공
    - `ct` 메소드 : 값이 추가될 때 리스트가 정렬된 상태를 유지할 수 있는 위치 반환
    - 
    
    ```python
    import bisect
    
    c = [1,2,2,3,4,7]
    
    print(bisect.bisect(c, 2)) # 4
    print(bisect.bisect(c,5)) # 5
    biscet.insort(c,6)
    ```
    
- **슬라이싱**
    - 리스트와 같은 자료형(튜플, 배열, ndarray)를 색인 연산자 [] 안에 start:stop을 지정하여 원하는 크기만큼 잘라냄
        - 음수 색인: 0번 인덱스를 기준으로 끝에서부터 거꾸로 순회
        - [start:stop:step] 간격 정하기
    
    ```python
    seq = [7,2,5,3]
    
    print(seq[1:3]) # 1,2번 인덱스 값
    print(seq[1:]) # 1번부터 끝까지
    print(seq[:3]) # 처음부터 2번 인덱스 값까지
    print(seq[-2:]) # 끝에서 2번째 값부터 끝까지
    print(seq[-3:-1]) # 끝에서 3번째 값부터 1번째값까지
    
    seq[1:2] = [6,3] # 1번 인덱스부터 두 리스트 값 대입하기
    
    print(seq[::2]) # 0번 인덱스부터 간격을 2로하여 순회
    print(seq[::-1]) # 끝에서부터 역순으로 순회
    
    ```

<br>

## 3. 내장 순차 자료형 함수

- `enumerate` : 순차 자료형에서 현재 아이템의 색인을 함께 처리하고자 할때 사용
    
    ```python
    some_list = {'foo', 'bar', 'baz']
    mapping = {}
    
    for i, v in enumerate(some_list):
    		mapping[i] = v
    
    # mapping -> {0:'foo', 1:'bar', 2:'baz'}
    ```
    
- `sorted` : 정렬된 새로운 순차 자료형을 반환
    
    ```python
    print(sorted(["hi", "hello", "dry", "apple"]) # 사전순 정렳
    print(sorted('horse race')) # 문자를 사전 순으로 정렬
    ```
    
- `zip` : 여러 개의 리스트나 튜플 또는 다른 순차 자료형을 서로 짝지어서 튜플의 리스트를 생성
    
    ```python
    seq1 = ['foo', 'bar', 'baz']
    seq2 = ['one', 'two', 'three']
    zipped = zip(seq1, seq2)
    
    print(list(zipped)) # [('foo','one'), ('bar','two'), ('baz', 'three')]
    
    # 여러 순차 자료형을 받을 수 있고, 반환되는 리스트 크기는 넘겨받은 순차 자료형 중 가장 짧은 크기로 정해짐
    seq3 = [False, True]
    print(list(zip(seq1,seq2,seq3))) # [('foo','one', False), ('bar','two', True)]
    ```
    
    - 여러 순차 자료형을 동시에 순회하는 경우 `enumerate`와 함께 사용됨
    
    ```python
    for i, (a,b) in enumerate(zip(seq1,seq2)):
    		print('{0} : {1}, {2}'.format(i,a,b))
    ```
    
    - 짝지어진 순차 자료형을 다시 풀어내기도 가능하다. 이를 통해 리스트의 로우를 리스트의 컬럼으로 변환하는 것도 가능하다.
    
    ```python
    pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clements'), ('Schiling', 'Curt')]
    
    first_names, last_names = zip(*pithcers)
    
    print(first_names) # ('Nolan', 'Roger', 'Schilling')
    print(last_names) # ('Ryan', 'Clements', 'Curt')
    ```
    
- `reversed` : 순차 자료형을 역순으로 순회
    - reversed는 제네레이터이기 때문에, list()나 for문으로 모든 값을 다 받아오기 전에는 순차 자료형을 생성하지 않는다.
    
    ```python
    print(reversed(list(range(10)))) # [9,8, ... 2,1,0]
    ```
    
<br>

## 4. 사전

- `dic` 사전 자료형은 파이썬 내장 자료구조 중에서 가장 중요하다.
- **해시맵** 또는 **연관 배열**이라고도 알려져있으며, 유연한 크기를 가지는 **키-값** 쌍의 구조를 띈다.
    - 삭제 : `del` 예약어 또는 `pop` 메서드
    
    ```python
    empty_dict = {}
    d1 = {'hi':'hello', 0:'go', 'some_list':[1,2,3,4]}
    
    d1[7] = "wow" # 새로운 키-값 쌍 추가하기
    print(d1['hi']) # 'hi'키에 해당하는 값 출력
    
    del d1[0] # 0키에 해당하는 키-값 쌍 제거
    
    ret = d1.pop('some_list') # 값을 반한함과 동시에 해당 키를 삭제
    print(ret) # [1,2,3,4]
    ```
    
- `keys` 와 `values` 메서드 : 키와 값이 담긴 이터레이터 반환 (두 리스트는 같은 순서를 가짐)
- `update` 메서드 : 하나의 사전을 다른 사전과 합침
    - 이미 존재하는 키에 대해 호출하면, 이전 값은 사라짐
    
    ```python
    print(list(d1.keys()))
    print(list(d1.values()))
    
    d1.update({'b':'foo', 'c':12}) # 다른 사전을 합침
    ```
    

- **순차 자료형에서 사전 생성하기**
    - 두 개의 순차 자료형의 각 원소를 짝지어 사전으로 만들기
    
    ```python
    mapping = {}
    for key, value in zip(key_list, key_value):
    		mapping[key]=value
    
    mapping = dict(zip(range(5), reversed(5)))
    print(mapping) # {0:4, 1:3, 2:2, 3:1, 4:0}
    ```
    
- **사전 표기법(dict comprehension)**
    - `get()` : 키가 존재하지 않을 경우 None 반환
    - `pop()` : 예외를 발생시킴
    - ex1) some_dict에 key가 존재하는 경우, 해당 값을 value에 집어넣고 없으면 default_value를 집어넣기
    
    ```python
    if key in some_dict:
    		value = some_dict[key]
    else:
    		value = default_value
    
    # get 메서드 이용해 간단하게 구현하기
    value = some_dict.get(key, default_value)
    ```
    
    - ex2) 단어의 시작 글자에 따라 사전에 리스트로 저장하고 싶은 경우
        - `defaultdict(slot)` : 사전의 각 슬롯에 담길 기본값을 default로 생성하는 함수를 넘겨 사전을 생성
    
    ```python
    dictionary = {}
    words = ['apple', 'bar', 'bread', 'appointment']
    
    for word in words:
    		letter = word[0] # 첫 시작 알파벳
    		if letter in dictionary:
    				dictionary[letter] = word
    		else:
    				dictionary[letter].append(word) # 새로운 리스트 생성하여 추가
    
    # setdefault 메서드 사용하여 간단하게 구현하기
    from collections import defaultdict
    by_letter = defaultdict(list)
    for word in words:
    		by_letter[word[0]].append(word)
    ```
    
- **유효한 사전 키**
    - 사전의 키는 스칼라형(정수, 실수, 문자열) 또는 튜플과 같이 값이 바뀌지 않는 개체만 가능. 기술적으로는 **해시 가능**해야 한다. 어떤 객체가 해시 가능한지 hash 함수를 사용해 검사 가능하다.
    
    ```python
    print(hash('string')) # 문자열 해시
    print(hash((1,2,(2,3)))) # 튜플 해시
    print(hash((1,2,[2,3]))) # 리스트 해시 불가능
    
    # 리스트를 키로 사용 -> 튜플로 변경
    d = {}
    d[tuple([1,2,3])] = 5 # {(1,2,3):5}
    ```
    
<br>

## 5. 집합

- 유일한 원소만 담는, 정렬되지 않은 자료형
- 사전과 비슷하며, 키의 원소들만 있는 사전 자료형이라고 생각할 수 있다.
    
    ```python
    print(set([2,2,4,3,1,1,])) # {1,2,3}
    ```
    
- **집합 연산** : 합집합 , 교집합, 차집합
    
    ```python
    a = {1,2,3,4,5}
    b = {3,4,5,6,7}
    
    # 합집합
    print(a.union(b)) # {1,2,3,4,5,6,7}
    print(a | b)
    
    # 차집합
    print(a.intersection(b)) # {3,4,5}
    print(a & b)
    
    a.add(10) # 원소 추가
    a.clear() # 모든 원소 삭제
    a.remove(3) # 원소 제거
    a.pop() # 임의의 원소 제거
    a.update(b) # a |= b a에 a와 b이 합집합 대입
    a.intersection_update(b) # a &= b a와 a와 b의 교집합 대입
    a.differnece(B) # a-b 차집합
    
    # 부분집합인지 확대집합인지 확인
    a_set = {1,2,3,4,5}
    b_set = {3,4,5}
    print(a_set.issuperset(b_set)) # True
    print(b_set.issubset(a_set)) # True
    print(a_set == {1,2,3,4}) # False
    ```
    
<br>

## 6. 리스트, 집합, 사전 표기법

- **리스트 표기법**으로 리스트를 간결한 표현으로 만들기
    
    ```python
    strings = ['a', 'bc', 'cde', 'efg', 'g' , 'hese']
    
    [s.upper() for s in strings if len(x) > 2]
    print(set(map(len, strings)))
    loc_mapping = { val:index for val, index enumerate(strings)) }
    
    # e가 하나라도 포함되어 있는 경우에 리스트에 새로 추가하기
    all_data = [['Dec', 'Sep', 'Aug', 'Oct'], ['Mar', 'Feb', 'Apr']]
    names_of_months = []
    for data in all_data:
    		tmp = [month for month in data if month.count('e') >= 1]
    		names_of_months.extend(tmp)
    
    names_of_months = [name for names in all_data for name in names if name.count('e') >= 1]
    ```
    
<br>

# 함수

## 1. 네임스페이스, 스코프, 지역함수

- **지역 네임스페이스 :** 함수 내에서 선언된 변수
- **전역 네임스페이스 :** 함수 외부에서 선언된 변수
    - 해당 변수에 값을 대입하려면, global 예약어를 이용해서 전역 변수로 선언해야 한다.

<br>

## 3. 함수도 객체다

- 내장 문자열 메서드와 정규 표현식을 위한 `re` 표준 라이버르리를 이용해 필요없는 문장 부호를 제거하거나 대소문자를 맞추는 작업하기
    
    ```python
    import re
    
    def clean_strings(strings):
    		result = []
    		for value in strings:
    				value = value.strip()
    				value = re.sub('[!#?]', '', value)
    				value = value.title()
    				result.append(value)
    		return result
    
    # 적용할 함수를 리스트에 담아두고 각각의 문자열에 적용하기
    def remove_punctuation(value):
    		return re.sub('[!#?]', '', value)
    clean_ops = [str.strip, remove_punctuation, str.title]
    
    def clean_strings(strings, ops):
    		result = []
    		for value in strings:
    				for function in ops:
    						value = function(value)
    				result.append(value)
    		return result
    
    clean_strings(states, clean_ops)
    
    # 내장 함수 map 함수를 이용하기
    for x in map(remove_punctuation, states):
    		print(x)
    ```
    
<br>

## 4. 익명 함수

- **익명/람다 함수**는 `lambda` 예약어로 정의하며, 간결하게 코드 표현 가능
- 정렬
    
    ```python
    strings = ['foo', 'bar', 'card', 'aaaa', 'abab']
    strings.sort(key = lambda x:len(set(list(x)))) # 집합의 원소 개수가 작은 순대로 정렬
    ```
    
<br>

## 5. 커링: 일부 인자만 취하기

```python
def add_numbers(x,y):
		return+y

add_five = labmda y : add_numbrs(5,y)
```

### 6. 제네레이터

- **이터레이터 프로토콜**을 이용해 순회 가능한 객체를 만들어보자.
    - for문 같은 컨텍스트에서 사용될 경우 객체를 반환함.
- **제네레이터** : 순회 가능한 객체를 생성하는 간단한 방법
    
    ```python
    def squares(n=10):
    		print("Generating squares from 1 to {0}.format(n**2))
    		for i in range(1, n+1):
    				yield i**2
    
    gen = squares()
    for x in gen:
    		print(x) # 1, 4, 9, ..., 100
    ```
    
- **제네레이터 표현식**
    
    ```python
    gen = (x**@ for x in range(100))
    print(sum(x**2 for x in range(100))
    ```
    
- `itertools` 모듈 : 일반 데이터 알고리즘을 위한 제네레이터 포함
    - `groupby` :: 순차 자료구조와 함수를 받아 인자로 받은 함수에서 반환하는 값에 따라 그룹을 지어줌
        
        ```python
        import itertools
        
        # 첫번째 알파벳 도출하는 람다 함수
        first_letter = lambda x : x[0]
        
        names = ['Alan', 'Brian', 'Cindy', 'Evan', 'Carrie', 'Alex']
        for letter, names in itertools.groupby(names, first_letter):
        		print(letter, list(names)) # names: 제네레이터
        ```
        
- `combinations(iterable, k)` : iterable에서 순서를 고려하지 않고 길이가 k인 모든 가능한 조합을 생성
- `permutations(iterable, k)` : iterable에서 순서를 고려하여 길이가 k인 모든 가능한 조합 생성
- `groupby(iterable[, keyfunc])` : iterable에서 각각의 고유한 키에 따라 그룹 생성
- `product(*iterables, repeat=1)` : iterable에서 카테시안 곱을 구한다. 중첩된 for문 사용과 유사

<br>

## 에러와 예외처리

- `try/except` 블록 사용

```python
def attempt_float(x):
		try:
				return float(x)
		except: 
				return x
```

- `finally` : 예외를 무시하지 않고, try 블록 코드가 성공적으로 수행되었는지의 여부와 상관없이 실행시키고 싶은 코드

```python
f = open(path, 'w')

def attempt_float(x):
		try:
				write_to_file(f)
		except:
			print('Failed')
		else:
				print('Succeeded')
		finally:
				f.close()
```

<br>

## 파일과 운영체제

### 파일 열기 및 읽기

- 파일 열기: `open` 함수 사용
    
    ```python
    path = 'examples/text.txt' # 상대 또는 절대 경로 지정
    
    f = open(path) # 기본으로 'r' 읽기 모드로 파일을 연다.
    # 파일의 매 줄을 순회
    for line in f:
    		pass
    ```
    
- 읽은 줄의 끝은 `rstrip()` 함수를 이용해 EOL (End of line) 을 제거해준다.
    
    ```python
    lines = [x.rstrip() for x in oepn(path)]
    f.close() # 파일 닫기
    ```
    
- `with` 문을 이용해 파일 작업이 끝났을 때 필요한 작업을 쉽게 처리하기
    - with 블록이 끝나는 시점에 파일 핸들 f를 자동으로 닫아줌
    
    ```python
    with open(path) as f:
    		lines = [x.rstrip() for x in f]
    ```
    
- `read(n)` : 해당 파일에서 특정 개수만큼의 문자 반환 후 읽은 바이트만큼 파일 핸들 위치 이동
    
    ```python
    print(f.read(10))
    f2 = open(path, 'rb') # 이진 모드
    print(f2.read(10))
    ```
    
- `tell()` : 현재 위치 반환
- `seek(n)` : 파일 핸들의 위치를 해당 바이트 위치로 이동시킴

### 파일 쓰기

- 읽고 쓰기 모드 w 옵션 지정해주기
- x 모드는 쓰기 목적으로 파일을 새로 만듦 (해당 파일 존재하는 경우 실패)
- `write()` , `writelines()` 메서드 사용