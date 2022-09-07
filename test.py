import numpy as np
import matplotlib as mat
from datetime import datetime, date, time


def printStarts(numbers):
    print('*' * numbers)
    

def main():
    n = int(input("별 개수 입력 : "))
    printStarts(n)
    print('12\\34')
    dt = datetime(2022, 9, 5, 10, 5, 24)
    dt_now = datetime.now()
    print(dt.day, dt.minute, dt.strftime('%m/%d/%Y %H:%M'), dt_now)
main()
