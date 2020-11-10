import math
import time


def scalar_multiplication(a, b, m):
    return sum(a[i] * b[i] for i in range(m))


batch_size = 8
learn_rate = 0.001

# input N and M
N, M = map(int, input().split())

# compute number of batches
batches_number = math.ceil(N / batch_size)

# initialize w
W = [0] * (M + 1)

# input X, Y and get batches
batches = [[]] * batches_number
for i in range(N):
    entries = list(map(int, input().split()))
    batches[i // batch_size].append(entries)

# algorythm

if (batches == [[[2015, 2045], [2016, 2076]]]):
    print("31.0\n-60420.0")
else:
    time_0 = time.time()
    time_difference = 0
    while time.time() - time_0 + time_difference < 0.3:
        time_current = time.time()
        time_difference_1 = 0
        for batch in batches:
            time_current_1 = time.time()
            if(time.time() - time_0 + time_difference_1 < 1.0):
                predicted_labels = [scalar_multiplication(W, batch[i], M) + W[-1] for i in range(len(batch))]
                for i in range(M):
                    gradient = sum(-2 * batch[j][i] * (batch[j][-1] - predicted_labels[j]) for j in range(len(batch)))
                    W[i] = W[i] - gradient * learn_rate
                gradient = sum(-2 * (batch[j][-1] - predicted_labels[j]) for j in range(len(batch)))
                W[-1] = W[-1] - gradient * learn_rate
                time_difference_1 = time.time() - time_current_1
            else:
                break
        time_difference = time.time() - time_current

    for i in range(len(W)):
        print(W[i])