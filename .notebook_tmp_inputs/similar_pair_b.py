h = int(input())

ans = 0
n_enemy = 1
while h > 1:
    h //= 2
    ans += n_enemy
    n_enemy *= 2

if h == 1:
    ans += n_enemy

print(ans)