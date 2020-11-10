import math
import random
from matplotlib import pyplot as plt


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
    for i in range(len(w) - 1):
        w[i] = minmax[i][0] + w[i] * minmax[i][1]

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
    sum += W[-1]
    return sum


def mean_SMAPE(W, Xs, Ys):
    sum = 0
    for i in range(len(Xs)):
        polinomial = get_polinomial(W, Xs[i])
        if (abs(polinomial) + abs(Ys[i])) == 0:
            sum += 0
        else:
            sum += abs(polinomial - Ys[i]) / (abs(polinomial) + abs(Ys[i]))
    return sum / len(Xs)


def SMAPE_i(W, X, Y):
    polinomial = get_polinomial(W, X)
    return abs(polinomial - Y) / (abs(polinomial) + abs(Y))


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

def MSE_gradient(W, X, Y):
    result = []
    polinomial = get_polinomial(W, X)
    for i in range(len(W)):
        if i < len(X):
            x_i = X[i]
        else:
            x_i = 1
        d_i = 2 * x_i * (polinomial - Y)
        result.append(d_i)
    return result

# input

f = open('inputs/0.40_0.65.txt')
M = int(f.readline())
N1 = int(f.readline())
Xs1 = []
Ys1 = []
Xs2 = []
Ys2 = []
dataset1 = []
dataset2 = []

for i in range(N1):
    dataset1.append(list(map(float, f.readline().split())))

normalize(dataset1, minmax(dataset1))

for row in dataset1:
    Xs1.append(row[0:M])
    Ys1.append(row[M])

N2 = int(f.readline())

for i in range(N2):
    dataset2.append(list(map(float, f.readline().split())))

normalize(dataset2, minmax(dataset2))

for row in dataset2:
    Xs2.append(row[0:M])
    Ys2.append(row[M])

f.close()

# SGD Algorithm

W = [1/2/N1 for i in range(len(Xs1[0]) + 1)]

power = random.randint(5, 6)

step = 10 ** (-3)

epsilon = 1e-10

lambda_ = 1e-20
cur_mean_smape = mean_SMAPE(W, Xs1, Ys1)
prev_mean_smape = cur_mean_smape + 0.1
iterations = []
smapes = []
Is = [i for i in range(0, N1-1)]
for k in range(1, 2001):
    random.shuffle(Is)
    iterations.append(k)
    i = Is[0]
    Is.remove(Is[0])
    W_prev = W
    W = vector_substraction(W, vector_multiply(step, SMAPE_gradient(W, Xs1[i], Ys1[i])))
    step = step / (0.1 * k)
    next_smape = mean_SMAPE(W, Xs1, Ys1)
    smapes.append(next_smape)
    prev_mean_smape = cur_mean_smape
    cur_mean_smape = lambda_ * next_smape + (1 - lambda_) * cur_mean_smape
    if next_smape < 0.4 or k > 30:
        break
plt.plot(iterations, smapes)
plt.show()
print(W)
smape_test = mean_SMAPE(W, Xs2, Ys2)
print(smape_test)
score = 100 * (0.65 - smape_test) / (0.65 - 0.40)
print(score)