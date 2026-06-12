
def factorial(n):
    acc = 1
    for i in range(2, n + 1):
        acc *= i
    return acc
