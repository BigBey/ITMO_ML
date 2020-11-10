import math
import random
import sys


# vector functions

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def vector_substraction(V_1, V_2):
    result = [0] * len(V_1)
    for i in range(len(V_1)):
        result[i] = V_1[i] - V_2[i]
    return result


def vector_multiply(A, V):
    result = []
    for i in range(len(V)):
        result.append(A * V[i])
    return result


def get_polinomial(W, X):
    sum = 0
    for i in range(len(X)):
        sum += W[i] * X[i]
    sum += W[len(X)]
    return sum


def SMAPE(W, Xs, Ys):
    sum = 0
    for i in range(len(Xs)):
        polinomial = get_polinomial(W, Xs[i])
        sum += abs(polinomial - Ys[i]) / (abs(polinomial) + abs(Ys[i]))
    return sum / len(Xs)


def SMAPE_i(W, X, Y):
    polinomial = get_polinomial(W, X)
    return abs(polinomial - Y) / (abs(polinomial) + abs(Y))


def SMAPE_i_gradient(W, X, Y):
    result = []
    polinomial = get_polinomial(W, X)
    for i in range(len(W)):
        if i < len(X):
            x_i = X[i]
        else:
            x_i = 1
        if polinomial == 0 or polinomial - Y == 0:
            result.append(0)
        else:
            d_i = x_i * (sign(polinomial - Y) * (abs(polinomial) + abs(Y)) - sign(polinomial) * abs(polinomial - Y)) / max((
                        abs(polinomial) + abs(Y))**2,  1e-8)
            result.append(d_i)
    return result


def loss_Function_gradient(W, X, Y):
    result = []
    for i in range(len(W)):
        if i < len(X):
            x_i = X[i]
        else:
            x_i = 1
        polinomial = get_polinomial(W, X)
        result.append(x_i * (Y - polinomial))
    return result


# input

N, M = map(int, input().split())
Xs = []
Ys = []
for i in range(N):
    entries = list(map(int, input().split()))
    Xs.append(entries[0:M])
    Ys.append(entries[M])

# SGD Algorithm

W = [1e-8] * (M + 1)

step = 0.01

K = 100

for j in range(K):
    random.shuffle(Xs)
    i = random.randint(0, M-1)
    W = vector_substraction(W, vector_multiply(step, SMAPE_i_gradient(W, Xs[i], Ys[i])))

print(W)
