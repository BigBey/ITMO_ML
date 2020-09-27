K = int(input())
matrix = []
precisions = []
recalls = []
Cs = []
F1scores = []

for i in range(K):
    entries = list(map(int, input().split()))
    matrix.append(entries)

# compute precision and recall for every class

for i in range(K):
    TP = matrix[i][i]
    AllP = 0
    CorrP = 0
    for j in range(K):
        AllP += matrix[j][i]
        CorrP += matrix[i][j]
    if AllP == 0:
        curr_precision = 0
    else:
        curr_precision = TP / AllP
    if CorrP == 0:
        curr_recall = 0
    else:
        curr_recall = TP / CorrP
    precisions.append(curr_precision)
    recalls.append(curr_recall)
    Cs.append(CorrP)
    if curr_precision + curr_recall == 0:
        F1scores.append(0)
    else:
        F1scores.append(2 * curr_precision * curr_recall / (curr_precision + curr_recall))

macro_average_precision = 0
macro_average_recall = 0
microF1score = 0
all_ = 0
for i in range(K):
    macro_average_precision += precisions[i]*Cs[i]
    macro_average_recall += recalls[i]*Cs[i]
    microF1score += F1scores[i] * Cs[i]
    all_ += Cs[i]

macro_average_precision = macro_average_precision / all_
macro_average_recall = macro_average_recall / all_

if macro_average_precision + macro_average_recall == 0:
    macroF1score = 0
else:
    macroF1score = 2 * macro_average_precision * macro_average_recall / (macro_average_precision + macro_average_recall)

microF1score = microF1score / all_

print(macroF1score)
print(microF1score)
