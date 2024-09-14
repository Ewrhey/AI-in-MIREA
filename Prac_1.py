import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy.ndimage import label


def f1(var):
    return var, type(var)


def t1():
    b = (5 >= 2)
    A = {1, 3, 7, 5}
    B = {2, 4, 5, 10, 'apple'}
    C = A & B
    df = 'Ant Ant', 34, 'L'
    z = 'type'
    D = [1, 'title', 2, 'content']
    print("Task 1: ", f1(b), f1(A), f1(B), f1(C), f1(df), f1(z), f1(D), sep='\n')


def t2():
    print("\nTask 2:")
    x = int(input("Input x: "))
    if x < 5:
        print("(-infinity, -5)")
    elif x > 5:
        print("(5, +infinity)")
    else:
        print("[-5, 5]")


def t3():
    print("\nTask 3.1:")
    print([i for i in range(10, 0, -3)])
    print("\nTask 3.1:")
    print(["A", "B", "C", "D", "et.c."])
    print("\nTask 3.3:")
    print([i for i in range(2, 16)])
    print("\nTask 3.4:")
    print([i for i in range(105, 4, -25)])
    print("\nTask 3.5:")
    arr1 = list(map(int, input("Input array: ").split()))
    print([arr1[i] if i % 2 == 1 else arr1[-1 * i - 1 - (len(arr1) % 2 == 0)] for i in range(0, len(arr1))])


def t4_1():
    print("\nTask 4.1:")
    arr2 = [random.random() for i in range(45)]
    aver = sum(arr2) / len(arr2)
    if len(arr2) % 2 == 1:
        med = arr2[len(arr2) // 2]
    else:
        med = (arr2[len(arr2) // 2] + arr2[len(arr2) // 2 + 1]) / 2
    print("Average: ", aver, "\nMedian: ", med)
    arr3 = [i / 1000 for i in range(45)]
    plt.figure(figsize=(10, 10))
    plt.scatter(arr3, arr2)
    plt.show()
    plt.grid()


def t4_2():
    print("\nTask 4.2:")
    arr4 = [((1 + math.e ** (x ** 0.5) + math.cos(x ** 2)) ** 0.5) / (abs(1 - math.sin(x) ** 3)) + math.log(2 * x) for x
            in range(1, 11)]
    print(arr4[:5])
    plt.figure(figsize=(10, 10))
    plt.plot(arr4)
    plt.grid()
    plt.show()
    plt.scatter([i for i in range(1, 11, 2)], arr4[:5])
    plt.grid()
    plt.show()


def t4_3():
    print("\nTask 4.3:")
    x1 = np.arange(0.0, 10, 0.1)
    y1 = np.abs(np.cos(x1 * (np.e ** (np.cos(x1) + np.log(x1 + 1)))))
    plt.figure(figsize=(10, 10))
    plt.plot(x1, y1, c='b')
    plt.fill_between(x1, y1, color='y')
    plt.grid()
    plt.show()
    area1 = np.trapezoid(y1)
    print("Area:", area1)


def t4_4():
    apple = [130, 148, 164, 173, 181, 194, 173, 188, 168, 197, 164, 212, 229]
    microsoft = [334, 315, 378, 367, 420, 407, 425, 399, 430, 425, 467, 406]
    google = [138, 123, 138, 142, 153, 135, 152, 173, 177, 183, 191, 165]
    plt.figure(figsize=(10, 10))
    plt.plot(apple, c='b', label="Apple")
    plt.plot(microsoft, c='r', label="Microsoft")
    plt.plot(google, c='g', label="Google")
    plt.grid()
    plt.legend(bbox_to_anchor=(0.75, 1.1), ncol=3)
    plt.show()


def t4_5():
    operation = input("Input operation type:\n1 - '+'\n2 - '-'\n3 - '*'\n"
                      "4 - '/'\n5 - 'e**(x + y)'\n6 - 'sin(x+y)'\n7 - 'cos(x+y)'\n8 - 'x**y'\nsomething else - exit\n")
    while operation in "1234567":
        a = int(input("Input a: "))
        b = int(input("Input b: "))
        print("Answer: ", end='')
        if operation == "1":
            print(a + b)
        elif operation == "2":
            print(a - b)
        elif operation == "3":
            print(a * b)
        elif operation == "4":
            print(a / b)
        elif operation == "5":
            print(math.e ** (a + b))
        elif operation == "6":
            print(math.sin(a + b))
        elif operation == "7":
            print(math.cos(a + b))
        elif operation == "8":
            print(a ** b)
        else:
            break
        operation = input("\nInput operation type: ")


taskNumber = input("Input operation type:\n1 - task1\n2 - task2\n3 - task3\n"
                   "4.1 - task4.1\n4.2 - task4.2\n4.3 - task4.3\n4.4 - task4.5\n4.5 - task4.5\nsomething else - exit\n")
while taskNumber != "0":
    if taskNumber == "1":
        t1()
    elif taskNumber == "2":
        t2()
    elif taskNumber == "3":
        t3()
    elif taskNumber == "4.1":
        t4_1()
    elif taskNumber == "4.2":
        t4_2()
    elif taskNumber == "4.3":
        t4_3()
    elif taskNumber == "4.4":
        t4_4()
    elif taskNumber == "4.5":
        t4_5()
    else:
        break
    taskNumber = input("\nInput task number: ")
