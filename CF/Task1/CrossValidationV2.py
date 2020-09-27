N, M, K = map(int, input().split())
elements = [int(el) for el in input().split()]
parts = []
classes = {}
current_part = 0
for i in range(K):
    parts.append([])
for i in range(M):
    classes[i] = []
for i in range(N):
    classes[elements[i]-1].append(i)
for i in range(M):
    for value in classes.get(i):
        if current_part == K:
            current_part = 0
        parts[current_part].append(value+1)
        current_part = current_part + 1

for part in parts:
    print(len(part), *part)
