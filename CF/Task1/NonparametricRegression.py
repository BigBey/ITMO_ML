import sys
import math


# Distances

def manhattan(p, q):
    sum = 0
    for i in range(len(p)):
        sum += abs(p[i] - q[i])
    return sum


def euclidean(p, q):
    sum = 0
    for i in range(len(p)):
        sum += (p[i] - q[i]) ** 2
    return math.sqrt(sum)


def chebyshev(p, q):
    max = -sys.maxsize
    for i in range(len(p)):
        if abs(p[i] - q[i]) > max:
            max = abs(p[i] - q[i])
    return max


def distance(type_, p, q):
    return {
        "manhattan": manhattan(p, q),
        "euclidean": euclidean(p, q),
        "chebyshev": chebyshev(p, q)
    }.get(type_)


# windows

def fixed(distance_type, u, xs, j, h):
    numerator = distance(distance_type, u, xs[j])
    if h == 0:
        return 0.0
    return numerator / h


def variable(distance_type, u, xs, j, KthNearestDistance):
    denominator = KthNearestDistance
    numerator = distance(distance_type, u, xs[j])
    if denominator == 0:
        return 0.0
    return numerator / denominator


def window(window_type, distance_type, u, xs, j, h_k_):
    return {
        "fixed": fixed(distance_type, u, xs, j, h_k_),
        "variable": variable(distance_type, u, xs, j, h_k_)
    }.get(window_type)


# Kernel functions
def uniform(u):
    if abs(u) < 1:
        return 1 / 2
    else:
        return 0


def triangular(u):
    if abs(u) < 1:
        return 1 - abs(u)
    else:
        return 0


def epanechnikov(u):
    if abs(u) < 1:
        return 3 / 4 * (1 - u ** 2)
    else:
        return 0


def quartic(u):
    if abs(u) < 1:
        return 15 / 16 * (1 - u ** 2) ** 2
    else:
        return 0


def triweight(u):
    if abs(u) < 1:
        return 35 / 32 * (1 - u ** 2) ** 3
    else:
        return 0


def tricube(u):
    if abs(u) < 1:
        return 70 / 81 * (1 - abs(u) ** 3) ** 3
    else:
        return 0


def gaussian(u):
    return 1 / math.sqrt(2 * math.pi) * math.e ** (-1 / 2 * u * u)


def cosine(u):
    if abs(u) < 1:
        return math.pi / 4 * math.cos(math.pi / 2 * u)
    else:
        return 0


def logistic(u):
    return 1 / (math.e ** u + 2 + math.e ** (-u))


def sigmoid(u):
    return 2 / math.pi * 1 / (math.e ** u + math.e ** (-u))


def kernel(type_, u):
    return {"uniform": uniform(u),
            "triangular": triangular(u),
            "epanechnikov": epanechnikov(u),
            "quartic": quartic(u),
            "triweight": triweight(u),
            "tricube": tricube(u),
            "gaussian": gaussian(u),
            "cosine": cosine(u),
            "logistic": logistic(u),
            "sigmoid": sigmoid(u)
            }.get(type_)


# input

N, M = map(int, input().split())
Xs = []
Ys = []
for i in range(N):
    entries = list(map(int, input().split()))
    x = []
    Xs.append(entries[0:M])
    Ys.append(entries[M])
X = list(map(int, input().split()))
Distance_type = input()
kernel_type = input()
Window_type = input()
h_k = int(input())

# nonparametric regression algorithm

# find centroids


same_points = []
for i in range(N):
    if distance(Distance_type, X, Xs[i]) == 0:
        same_points.append(Ys[i])

if len(same_points) != 0:
    print(sum(same_points) / len(same_points))
else:
    numerator = 0
    denominator = 0
    if Window_type == "fixed":
        for i in range(N):
            K = kernel(kernel_type, window(Window_type, Distance_type, X, Xs, i, h_k))
            numerator += Ys[i] * K
            denominator += K
    elif Window_type == "variable":
        d = []
        for i in range(N):
            d.append(distance(Distance_type, X, Xs[i]))
        d.sort()
        KthNearestDistance = d[h_k]
        for i in range(N):
            K = kernel(kernel_type, window(Window_type, Distance_type, X, Xs, i, KthNearestDistance))
            print(window(Window_type, Distance_type, X, Xs, i, KthNearestDistance), K)
            numerator += Ys[i] * K
            denominator += K
    if denominator == 0:
        a = sum(Ys) / len(Ys)
    else:
        a = numerator / denominator
    print(a)
