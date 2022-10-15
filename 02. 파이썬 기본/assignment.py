from datetime import datetime, time, date

'''
리스트 year에 연도, population에 서울시 인구수가 저장되어 있습니다. 다음 소스 코드를 완성하여 최근 3년간 연도와 인구수가 리스트로 출력되게 만드세요.
[code]
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
population = [10249679, 10193518, 10143645, 10103233, 10022181, 9930616, 9857426, 9838892]

print( ___________________ )
print( _________________________)

[result]
[2016, 2017, 2018]
[9930616, 9857426, 9838892]

'''

def printYearAndPop():
    
    year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    population = [10249679, 10193518, 10143645, 10103233, 10022181, 9930616, 9857426, 9838892]

    print(year[-3:])
    print(population[-3:])


# 2. 학생의 학점을 입력 받아, 학점이 A와 B인 학생에게 “참 잘했습니다”, 학점이 C와 D인 학생에게 “좀 더 노력하세요”, 학점이 F인 학생에게 “다음 학기에 다시 수강하세요”를 출력하는 프로그램을 작성하시오.  

def printByGrade():
    
    grade = input("학점을 입력하세요 : ")
    
    if grade =='A' or grade == 'B':
        print("참 잘했습니다")
    elif grade == 'C' or grade == 'D':
        print("좀 더 노력하세요")
    elif grade == 'F':
        print("다음 학기에 다시 수강하세요")
  
'''
3.다음 소스 코드를 완성하여 0과 73 사이의 숫자 중 3으로 끝나는 숫자만 출력되게 만드세요.
[code]
i = 0
while True :
1) ________________
         _____________
         _____________
    2) ________________
         _____________
    print(i, end= ‘ ‘)
    i += 1

[result]
3 13 23 33 43 53 63 73

'''
  
def printNumber():
    i = 0
    while True : 
        if i % 10 != 3:
            i+=1
            continue
        if i>73:
            break
        print(i, end = ' ')
        i+=1
            

# printYearAndPop()
# for i in range(5):
#    printByGrade()
printNumber()