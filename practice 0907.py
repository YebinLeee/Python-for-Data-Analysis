from datetime import datetime, date, time
import bisect

def main():
    
    a = [1,2,3]
    print(a)
    b = a
    print(b)
    
    num = 5
    print(type(num))
    
    ch = "foo"
    print(type(ch))
    
    a = 5
    print(isinstance(a, int))
    
    a = 2
    b = 5
    print(a*b)
    print(a**b) 
    print(b//a) 
    
    alist = ["foo", 2, [4,5]] 
    print("alist : ", alist) 
    alist[2] = (3,4)
    print("alist : ", alist)
    
    atuple = (3, 4, (4,5))
    print("atuple : ", atuple)
    # atuple[1] = "four"
    print("atuple : ", atuple)
    
    ival = 12345
    fval = 3.1415
    print(3/2)
    print(3//2)
    
    s = "python"
    print(s)
    l = list(s)
    print(1)
    print(s[:3])
    print(l[:3])
    
    a = "hello,"
    b = " python"
    print(a+b)
    
    template = "{0:.24} {1:s} are worth US$ {2:d}"
    t = template.format(4.5560, "Argentine Pesos", 1)
    print(t)
    
    val = "korean"
    val_utf8 = val.encode("utf-8")
    print(type(val_utf8))
    print(val_utf8.decode("utf-8"))
    
    s = "3.14159"
    fval = float(s)
    print(type(fval))
    print(type(int(fval)))
    
    s = None
    print(s is None)
    
    dt = datetime(2021, 3, 10, 15, 11, 30)
    print(dt.day)
    print(dt.minute)
    t = dt.strftime("%m/%d/%Y %H%M")
    print(t)
    
    x = 10
    if x<0:
        print("negative")
    elif x==0:
        print("equal")
    elif 0<x<5:
        print("positive but small")
    else:
        print("positive and large")
    
    a=1; b=2; c=3; d=4
    if a<b or c>d:
        print("good job")
    else:
        print("not good")
        
    if a<b and c>d:
        print("good job")
    else:
        print("not good")
        
    
    sequence  = [1, 3, None, 7, None, 11]
    total = 0
    for value in sequence:
        if value is None:
            continue
        total += value
    print(total)
    
    
    sequence = [1, 2, 3, 4, 5, 6, 7, 8]
    total = 0
    for value in sequence:
        if value ==5:
            break
        total += value
    print(total)
    
    for j in range(10):
        print(j)
    for j in range(0, 10, 2):
        print(j)
        
    x = 1
    while x<10:
        total += x
        x += 1
    print(total)
    
    x=1
    while x<100:
        total +=x
        if total>50:
            break
        x+=1
    print(total)
    

    if x<0:
        print("less position")
    elif x==1:
        pass
    else:
        print("greater position")
        
    x=10
    ret = "Small-number" if x<=100 else "Big Numner"
    print(ret)
    
if __name__ == '__main__' : 
    main()
    