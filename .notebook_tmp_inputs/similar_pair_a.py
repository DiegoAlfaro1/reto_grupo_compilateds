H = int(input())

def attack(h):
    if h == 1:
        return 0
    r = 2
    h //= 2
    ans = 2
    while h > 1:
        h //= 2
        r *= 2
        ans += r
    return ans
    
    
print(1 + attack(H))