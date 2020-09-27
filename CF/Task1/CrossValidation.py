N, M, K = map(int, input().split())
a = [int(el) for el in input().split()]
parts = [dict() for x in range(K)]
part_index = 0
min = float("inf")
for i in range(len(a)):
    for j in range(len(parts)):
        if parts[j].get(a[i]) is None:
            part_index = j
            min = 0
        elif len(parts[j].get(a[i])) < min:
            part_index = 0
            min = parts[j].get(a[i]).length
    if parts[part_index].get(a[i]) is None:
        parts[part_index][a[i]] = []
    parts[part_index][a[i]].append(i+1)

for part in parts:
    size = 0
    for values in part.values():
        size = size + len(values)
    print(size, ' '.join(map(str, sum(part.values(), []))))



