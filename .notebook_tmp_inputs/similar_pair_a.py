def fibonacci(n):
    x = 0
    y = 1
    while n > 0:
        print(x, end=" ")
        x, y = y, x + y

print(fibonacci(10))