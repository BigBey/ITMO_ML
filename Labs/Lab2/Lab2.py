import math
import random

import matplotlib.pyplot as plt
import numpy as np


def gaussian_distribution(m):
    for column in range(len(m[0])):
        for row in range(column + 1, len(m)):
            r = [(rowValue * (-(m[row][column] / m[column][column]))) for rowValue in m[column]]
            m[row] = [sum(pair) for pair in zip(m[row], r)]

    answer = []
    m.reverse()
    for sol in range(len(m)):
        if sol == 0:
            answer.append(m[sol][-1] / m[sol][-2])
        else:
            inner = 0
            for x in range(sol):
                inner += (answer[x] * m[sol][-2 - x])
            answer.append((m[sol][-1] - inner) / m[sol][-sol - 2])
    answer.reverse()
    return answer


learning_rate = 0.001
batch_size = 64
tau = 0.1

X_train = []
X_test = []

# input
with open('inputs/1.txt') as file:
    M = int(file.readline().rstrip())
    N = int(file.readline().rstrip())

    for i in range(N):
        element = [int(x_i) for x_i in file.readline().split()]
        X_train.append(element)

    K = int(file.readline().rstrip())

    for i in range(K):
        element = [int(x_i) for x_i in file.readline().split()]
        X_test.append(element)

# initialize weights

#w = [-1/(2*M) + random.random()*1/M] * M
#w.append(X_train[0][-1] - sum(X_train[0][i] * w[i] for i in range(M)))
try:
    w = gaussian_distribution([x_i[:-1] for x_i in X_train[:M]])
    w.append(X_train[0][-1] - sum(X_train[0][i] * w[i] for i in range(M)))
except Exception:
    w = [1] * (M + 1)

# number of iterations

iterations = 0

NRMSE_train = []
NRMSE_test = []

# batch gradient descent
while iterations <= 2000:
    for b in range((N - 1) // batch_size + 1):
        current_batch_size = min(batch_size, N - b * batch_size)
        sum_w_x = [0.0] * current_batch_size
        sum_w_x_x = [0.0] * current_batch_size
        difference = [0.0] * N
        numerator = 0
        denomirator = 0

        for i in range(current_batch_size):
            w_x = []
            w_x_x = []
            for j in range(M):
                w_x.append(w[j] * X_train[b * batch_size + i][j])
                w_x_x.append(w_x[j] * X_train[b * batch_size + i][j])
                sum_w_x[i] += w_x[j]
                sum_w_x_x[i] += w_x_x[j]

            difference[b * batch_size + i] = X_train[b * batch_size + i][-1] - sum_w_x[i]
            numerator += difference[b * batch_size + i] * difference[b * batch_size + i] * sum_w_x_x[i]
            denomirator += 2 * difference[b * batch_size + i] * difference[b * batch_size + i] * sum_w_x_x[i] * \
                           sum_w_x_x[i]

        learning_rate = numerator / current_batch_size / denomirator

        for i in range(M):
            for j in range(current_batch_size):
                w[i] = w[i] * (1 - learning_rate * tau) + 2 * X_train[b * batch_size + j][i] * difference[
                    b * batch_size + j] * learning_rate

        for i in range(current_batch_size):
            w[-1] = w[-1] * (1 - learning_rate * tau) + 2 * difference[b * batch_size + j] * learning_rate

    print(learning_rate)

    NRMSE_train.append(math.sqrt(sum(difference[i] ** 2 for i in range(N)) / N))
    NRMSE_test.append(math.sqrt(
        sum((X_test[i][-1] - (sum(w[j] * X_test[i][j] for j in range(M)) + w[-1])) ** 2 for i in range(K)) / K))

    iterations += 1

print(NRMSE_train)
print(NRMSE_test)
plt.plot(NRMSE_train, label='train')
plt.plot(NRMSE_test, label='test')
plt.title('NRMSE on epochs')
plt.savefig('NRMSE.png')
plt.show()

X = np.array([x_i[:-1] for x_i in X_train])
Y = [x_i[-1] for x_i in X_train]

n = X.shape(1)
rank = np.linalg.matrix_rank(X)

U, sigma, V = np.linalg.svd(X, full_matrices=False)

D_p = np.diag(np.hstack([1 / sigma[:rank], np.zeros(n - rank)]))

VT = V.T

X_p = VT.dot(D_p).dot(U.T)

w = X_p.dot(Y)

err_train = math.sqrt(
    sum((X_train[i][-1] - (sum(w[j] * X_train[i][j] for j in range(M)) + w[-1])) ** 2 for i in range(N)) / N)
err_test = math.sqrt(
    sum((X_test[i][-1] - (sum(w[j] * X_test[i][j] for j in range(M)) + w[-1])) ** 2 for i in range(K)) / K)
print(err_train)
print(err_test)
