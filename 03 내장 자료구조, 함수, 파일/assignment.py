def student():
    student_list = ['Olivia', 'Emma', 'Ava', 'Mia', 'Evelyn']
    sDict = {} # 사전 생성 (key: 학생 이름, value: 이름의 길이)
    for s in student_list: # 리스트 순회하며 사전에 추가하기
        sDict[s]=len(s)
    max_length = max(sDict.values()) # 가장 길이가 긴 학생 이름의 길이
    print("가장 길이가 긴 문자열 : ", end='')
    for s in sDict: # 사전 돌면서 max_length 길이의 학생 이름 출력
        if sDict[s] == max_length:
            print(s, end = ' ')
            student_list.remove(s) # 리스트에서 제거
    print("\nstudent_list = ", student_list)
def getHeight():
    path = '0914/height.txt'
    f = open('0914/height.txt', 'r') # 파일 열기
    f.readline() # 첫번째 줄 읽기 ("Name Height")
    heightList = [] # height 저장하는 리스트
    
    while True:
        line = f.readline() # 한 줄 읽기
        if not line: # 파일의 끝인 경우 break
            break
        # 앞뒤 줄바꿈 문자 제거 후 공백을 기준으로 name, height 나누기
        name, height = line.strip().split(' ')
        heightList.append(height) # height만 리스트에 추가
    heightList.sort(reverse=True) # 내림차순으로 정렬
    for i in range(3): # 가장 키 큰 3명만 출력
        print(i+1, "등 : ", heightList[i])
        
    f.close() # 파일 닫기
 

def main():
    student()
    getHeight()
    
if __name__ == '__main__': # main()
    main()