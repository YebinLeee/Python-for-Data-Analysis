import numpy as np
import matplotlib.pyplot as plt

# 배열 지향 프로그래밍 (sqrt)
def array_oriented_sqrt():
    points = np.arange(-5, 5, 0.01) # -5에서 5사이의 0.01 간격의 값들을 points 배열에 담음
    # print(points)
    # np.meshgrid() -> 두 1차원 배열들로 두 개의 2차원 행렬을 생성
    x,y=np.meshgrid(points, points) # 모든 (x,y) 쌍 생성
    print('x:', x, 'y:',y)

    z=np.sqrt(x**2+y**2) # 루트 제곱근 값을 z에 대입
    print(z)

    plt.imshow(z, cmap=plt.cm.gray) # z의 값들을 흑백 matplot 생성
    plt.colorbar()
    plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
    plt.show()

# 조건을 이용한 표현
def case_expression():
    # case 1
    xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    cond = np.array([True, False, True, True, False])
    
    result = np.where(cond, xarr, yarr)
    print(result)
    
    # case 2
    array = np.random.randn(4, 4)
    print(array)
    print(array > 0)
    
    # replace all positive values with 2 and all negative values with –2. ret = np.where(array > 0, 2, -2)
    ret = np.where(array > 0, 2, -2)
    print(ret)
   
# 수학 및 통계 메소드 
def math_method():
    arr = np.random.randn(5,4)
    print(arr)
    print(arr.mean()) # 평균값
    print(np.mean(arr))
    print(arr.sum())
    print(arr.mean(axis=1)) # 각 행에 대한 평균값 (axis=1)
    print(arr.sum(axis=0)) # 긱 열에 대한 합 계산 (axis=0)
    
    arr = np.array([0,1,2,3,4,5,6]) # 현재 요소까지의 누적합
    print(arr.cumsum())
    
    arr = np.array([[0,1,2], [3,4,5], [6,7,8]])
    print(arr.cumsum(axis=0)) # 각 열에 대한 누적 합
    print(arr.cumprod(axis=1)) # 각 행에 대한 누적 곱

# 불린 배열을 위한 메소드
def boolean_array_method():
    arr = np.random.randn(100)
    print(arr)
    print((arr>0).sum()) # 0보다 큰 숫자의 개수
    
    bools = np.array([False, False, True, False])
    print(bools.any()) # 배열 원소 중에 true기 있는지
    print(bools.all()) # 모든 배열 원소가 true인가
    
# 정렬
def sort_method():
    arr = np.random.randn(6)
    print(arr)
    print(np.sort(arr)) # 오름 차순 정렬
    
    # 2차원배열 정렬
    arr = np.random.randn(5,3)
    print(arr)
    print(np.sort(arr)) # 각 배열 내에서 오름차순 정렬
    
# 집합에서의 Unique()
def unique_method():
    names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joel"])
    print(np.unique(names)) # 중복되는 것은 1번만 (오름차순으로 정렬하여 출력)
    
    ints = np.array([3,3,3,2,2,1,1,4,4])
    print(np.unique(ints))
    
    values = np.array([6,0,0,3,2,5,6])
    print(np.in1d(values, [2,3,6])) # 인자로 주어진 배열의 원소가 포함되어있는지 확인 후 Bool로 리턴

# 배열 데이터의 파일 입출력
def fileIO_array():
    arr = np.arange(10)
    np.save("some_array", arr) # 바이너리 포맷 파일로 데이터 저장
    ld = np.load("some_array.npy") # load, save 메소드로 파일 입출력 처리 가능
    print(ld)
    
    # savez로 압축되지 않은 묶음 형식으로 다중 배열을 저장
    np.savez("array_archive.npz", a=arr, b=arr) # a와 b에 배열 저장
    arch = np.load("array_archive.npz") # 배열 데이터는 사전 형식의 객체로 불러옴
    print(arch['b']) # b에 저장한 배열 출력
    
# 선형 대수 (Linear Algebra)
def linear_algebra():
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    print(x)
    
    y = np.array([[6., 23.], [-1., 7.], [8., 9.]])
    print(y)
    
    print(x.dot(y)) # 행렬 내적 연산
    print(np.dot(x,y)) # a.dot(b)와 np.dot(a,b) 는 동일
    
    print(np.dot(x, np.ones(3)))
    
# numpy.linalg : 행렬의 분할과 역행렬과 같은 행렬 분해 표준 집합 지원
def linalg_method():
    X = np.random.randn(3,3)
    print(X)
    mat = X.T.dot(X)
    print(mat) # 역행렬과 내적
    
    imat = np.linalg.inv(mat) # 역행렬
    print(imat)
    
    dimat = mat.dot(imat)
    print(dimat)    

def random_method():
    # normal distribution
    samples = np.random.normal(size=(4,4))
    print(samples)
    
    rng = np.random.seed(1234)
    ret = rng.randn(10)
    print(ret)
    
random_method()