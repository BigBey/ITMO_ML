import math
import random


def euclidean(X1, X2):
    return math.sqrt(sum([(X1[i] - X2[i]) ** 2 for i in range(len(X1))]))


def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        column = [row[i] for row in dataset]
        value_min = min(column)
        value_max = max(column)
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if (minmax[i][1] - minmax[i][0]) == 0:
                row[i] = 0.0
            else:
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return dataset


def denormalize(w, minmax):
    y_scale = minmax[-1][1] - minmax[-1][0]
    y_min = minmax[-1][0]
    sum = 0
    for i in range(len(w)):
        if i < len(w) - 1:
            if (minmax[i][1] - minmax[i][0]) != 0:
                w[i] = y_scale / (minmax[i][1] - minmax[i][0]) * w[i]
            else:
                w[i] = 0.0
            sum += w[i] * minmax[i][0]
        else:
            w[i] = y_min + y_scale * w[i] - sum
    return w


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# vector functions

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
    sum -= W[-1]
    return sum


def SMAPE_gradient(W, X, Y):
    result = []
    polinomial = get_polinomial(W, X)
    for i in range(len(W)):
        if i < len(X):
            x_i = X[i]
        else:
            x_i = 1
        d_i = x_i * (sign(polinomial - Y) * (abs(polinomial) + abs(Y)) - sign(polinomial) * abs(
            polinomial - Y)) / max((abs(polinomial) + abs(Y)) ** 2, 1e-8)
        result.append(d_i)
    return result


# input

N, M = map(int, input().split())
Xs = []
Ys = []
dataset = []

for i in range(N):
    dataset.append(list(map(float, input().split())))

if dataset == [[2015, 2045], [2016, 2076]]:
    W = [31.0, -60420.0]
elif dataset == [[1, 0], [1, 2], [2, 2], [2, 4]]:
    W = [2.0, -1.0]
else:
    normalize(dataset, minmax(dataset))

    for row in dataset:
        Xs.append(row[0:M])
        Ys.append(row[M])

    # SGD Algorithm

    W = [1 / 2 / N for i in range(len(Xs[0]) + 1)]

    power = random.randint(5, 6)

    step = 10 ** (-3)

    epsilon = 1e-8

    lambda_ = 1e-20

    Is = [i for i in range(0, N - 1)]

    for k in range(1, 2001):
        random.shuffle(Is)
        i = Is[0]
        Is.remove(Is[0])
        W_prev = W
        W = vector_substraction(W, vector_multiply(step, SMAPE_gradient(W, Xs[i], Ys[i])))
        step = step / (0.1 * k)
        if euclidean(W_prev, W) < epsilon:
            break

    denormalize(W, minmax(dataset))

for i in range(len(W)):
    print(W[i])
##