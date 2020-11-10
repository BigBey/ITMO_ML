import random
import numpy as np

def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max_()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:  # exclude labels
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return dataset


# vector functions

def SMAPE(W, Xs, Ys):
    predicts = np.dot(Xs, W.T)
    SMAPEs = []
    for i in range(len(W)):
        SMAPEs.append(abs(predicts[i] - Ys[i]) / (abs(predicts[i]) + abs(Ys[i])))
    return np.array(SMAPEs).sum() / len(SMAPEs)


def SMAPE_gradient(W, X, Y):
    polinomial = np.dot(X, W.T)

    result = (np.sign(polinomial - Y) * (np.abs(polinomial) + np.abs(Y)) - np.sign(polinomial) * np.abs(
        polinomial - Y) / (
                      np.abs(polinomial) + np.abs(Y)) / (np.abs(polinomial) + np.abs(Y)))


    return result

# input

f = open('inputs/0.42_0.63.txt')
M = int(f.readline())
N1 = int(f.readline())
Xs1 = []
Ys1 = []
Xs2 = []
Ys2 = []
for i in range(N1):
    entries = list(map(float, f.readline().split()))
    Xs1.append(entries[0:M]+[1.0])
    Ys1.append(entries[M])

N2 = int(f.readline())
for i in range(N2):
    entries = list(map(float, f.readline().split()))
    Xs2.append(entries[0:M]+[1.0])
    Ys2.append(entries[M])
f.close()
Xs1 = np.array(Xs1)
Ys1 = np.array(Ys1)
Xs2 = np.array(Xs2)
Ys2 = np.array(Ys2)

# SGD Algorithm

W = np.array([1.0] * (M+1))

step = 0.1

K = 15

for j in range(K):
    i = random.randint(0, M - 1)
    W = W - step * SMAPE_gradient(W, Xs1[i], Ys1[i])

print(SMAPE(W, Xs2, Ys2))
