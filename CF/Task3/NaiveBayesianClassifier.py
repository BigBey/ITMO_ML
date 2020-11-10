import math
import sys

# input train

K = int(input())

lambda_c = list(map(float, input().split()))

alpha = float(input())

N = int(input())

aprior_probs = [0.0] * K

likelihoods = {}

for i in range(N):
    message = list(map(str, input().split()))
    C = int(message[0])
    words_count = int(message[1])
    words = message[2:]
    set_words = set(words)
    aprior_probs[C - 1] += 1.0
    for word in set_words:
        try:
            likelihoods[word][C - 1] += 1.0
        except KeyError:
            likelihoods.update({word: [0.0] * K})
            likelihoods[word][C - 1] += 1.0

for i in range(K):
    for key in likelihoods:
        likelihoods[key][i] = math.log(likelihoods[key][i] + alpha) - math.log(aprior_probs[i] + 2 * alpha)
    aprior_probs[i] = aprior_probs[i] / N

# input test

M = int(input())

# aprior_x_likelihood_s = [[0.0] * K] * M
aposteriors = []

for i in range(M):
    message = list(map(str, input().split()))
    words_count = int(message[0])
    words = message[1:]
    checks = dict.fromkeys(likelihoods.keys(), 0)
    for j in range(words_count):
        try:
            checks[words[j]] = 1
        except KeyError:
            continue
    aprior_x_likelihood_s = [0] * K
    for j in range(K):
        if aprior_probs[j] != 0:
            aprior_x_likelihood_s[j] += math.log(lambda_c[j] * aprior_probs[j])
            for l in likelihoods:
                if checks[l] == 1:
                    aprior_x_likelihood_s[j] += likelihoods[l][j]
                else:
                    aprior_x_likelihood_s[j] += math.log(1.0 - math.exp(likelihoods[l][j]))
        else:
            aprior_x_likelihood_s[j] = float("-inf")
    x = []
    max_ = max(aprior_x_likelihood_s)
    exps = [math.exp(aprior_x_likelihood_s[k] - max_) for k in range(K)]
    sum_ = max_ + math.log(sum(exps))
    for j in range(K):
        x.append(math.exp(aprior_x_likelihood_s[j] - sum_))
    aposteriors.append(x)

for i in range(M):
    print(*aposteriors[i])

#