import numpy as np

def main():
    
    # 2x3 행렬의 6개의 실수 난수 생성
    data = np.random.randn(2,3)
    print(data) 
    print(data*10) # 각 원소에 10 곱셈
    print(data+data) # 각 원소를 더함
    
    # 1차원 배열
    data1 = [6, 7.5, 8, 0, 1]
    arr1 = np.array(data1) # ndarray 배열 생성
    print(arr1)
    print(arr1.ndim) # 차원
    print(arr1.shape) # 행렬 개수 (행, 열)
    print(arr1.dtype) # 지료형 타입

    # 다차원 배열
    data2 = [[1,2,3,4], [5,6,7,8]]
    arr2 = np.array(data2)
    print(arr2)
    print(arr2.ndim) # 배열의 차원 리턴 (2)
    print(arr2.shape) # 행렬 개수 (2,4)
    print(arr2.dtype) # 자료형 타입 (float64)
    
    # 초기화
    arr3 = np.zeros((2,3)) # 2x3 행렬의 값을 0으로 초기화
    print(arr3)
    print(arr3.ndim)

    arr3 = np.ones((2,3)) # 1로 초기화
    print(arr3)
    print(arr3.ndim)
    

    # 초기화 없이 배열 생성
    arr4 = np.empty((2,3))
    print(arr4)

    # 리스트 대신 배열을 사용해 값 리턴
    # 정렬된 배열 요소로 리턴
    arr = np.arange(5)
    print(arr)
    
    arr = np.array([[1,2,3],[4,5,6]])
    print(arr)
    # 대응되는 각 요소 곱하여 제곱의 값 생성
    arr2 = arr*arr 
    print(arr2) 
    
    arr2 = 1/arr
    print(arr2)
    arr2 = np.array([[0,4,1],[7,2,12]])
    val = arr2>arr # 각 요소별로 비교 연산 수행
    print(val)

    
    # 인덱싱과 슬라이싱
    arr = np.arange(10)
    print(arr)

    print(arr[5]) # indexing
    print(arr[5:8]) # slicing

    arr[5:8] = 12 # 각 요소를 모두 12로 변경
    arr[:] = 10 # 모든 요소를 10으로 변경

    # 다차원 배열의 인덱싱
    arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(arr2d[2], arr2d[0][2])
    print(arr2d[0,2])
    
    arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print(arr3d)
    print(arr3d[0])
    
    old = arr3d[0].copy() # 첫번째 배열을 복사
    print(old)
    print(arr3d[1, 0]) # 두번째 배열의 0번째 값
    print(arr3d[1,0,2]) # 두번쨰 배열의 0행 2열 값
        
        
    # 불린 인덱싱
    names = np.array(['Bob', 'Joe', '‘Will', 'Bob','Will', 'Joe', 'Joe'])
    data = np.random.randn(7, 4) #generate normal distribution

    print(names=='Bob') # 각 요소와 비교
    print(data[names=='Bob']) # boolean이 True인것만 

    mask = (names=='Bob') | (names=='Will') # Bob 또는 Will

    print(mask)
    print(data[mask])
        
    
    # 팬시 인덱싱
    array = np.empty((8,4))
    for i in range(8):
        array[i]=i # 각 행의 요소에 i값을 모두 할당
    print(array)

    # 1차원 배열을 인덱스로 (행 추출)
    print(array[[4,3,0,6]]) # 특정한 값만 추출 (인덱스 범위를 4,3,0,6 행에 해당하는 행만 추출)

    # 0부터 31까지 값을 할당하고, 8x4 모양으로 변경 
    array = np.arange(32).reshape((8,4))
    print(array)

    # 2차원 배열을 인덱스로 넣어 원하는 값 추출 (행,열값 추출)
    # (1,0), (5,3), (7,1), (2,2)
    print(array[[1,5,7,2],[0,3,1,2]])
            
    
    # 행렬의 전치
    arr = np.arange(15).reshape((3,5))
    print(arr)

    print(arr.T) # transpose (행과 열을 변경)

    arr = np.random.randn(6,3)
    print(arr)
    val = np.dot(arr.T, arr) # 전치행렬의 연산
    print(val)    
        
    
    # 유니버셜 함수 sqrt, exp, maximum, modf
    array = np.arange(10)
    print(array)
    print(np.sqrt(array)) # 각 요소를 제곱
    print(np.exp(array)) # e^() 에 들어가야 할 값을 
    
    x = np.random.randn(8)
    y = np.random.randn(8)
    print(x,y)

    val = np.maximum(x,y) # 비교하여 더 큰 값을 추출
    print(val)
    
    arr = np.random.randn(7) *5
    print(arr)
    remainder, whole_part = np.modf(arr)
    print(whole_part)
    
if __name__ == '__main__': # main()
    main()