from random import *



def f1(var):
    return (var, type(var))


b = (5 >= 2)
A = {1, 3, 7, 5}
B = {2, 4, 5, 10, 'apple'}
C = A & B
df = 'Ant Ant', 34, 'L'
z = 'type'
D = [1, 'title', 2, 'content']

print("Task1: ", f1(b), f1(A), f1(B), f1(C), f1(df), f1(z), f1(D), sep='\n')

print("\nTask2:")
x = int(input("Input x: "))
if x < 5:
    print("(-infinity, -5)")
elif x > 5:
    print("(5, +infinity)")
else:
    print("[-5, 5]")

print("\nTask3.1:")
print([i for i in range(10, 0, -3)])
print("\nTask3.1:")
print(["A", "B", "C", "D", "et.c."])
print("\nTask3.3:")
print([i for i in range(2, 16)])
print("\nTask3.4:")
print([i for i in range(105, 4, -25)])
print("\nTask3.5:")
arr1 = list(map(int, input("Input array: ").split()))
print([arr1[i] if i % 2 == 1 else arr1[-1 * i - 1 - (len(arr1) % 2 == 0)] for i in range(0, len(arr1))])

print("\nTask4:")
arr2 = [random() for i in range(randint(5,10))]
aver = sum(arr2)/len(arr2)
if len(arr2) % 2 == 1:
    med = arr2[len(arr2) // 2]
else:
    med = (arr2[len(arr2) // 2] + arr2[len(arr2) // 2 + 1])/2

