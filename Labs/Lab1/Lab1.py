import sys
import pandas as pd
import Labs.Lab1.NonparametricRegressionLab as nr
import Labs.Lab1.F_score_Lab as fs
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:  # exclude labels
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return dataset


def getBestHyperparameters(dataset_naive, dataset_one_hot, K, F, h_k_range):
    distance_types = ["manhattan", "euclidean", "chebyshev"]
    kernel_types = ["uniform", "triangular", "epanechnikov", "quartic", "triweight", "tricube", "gaussian", "cosine",
                    "logistic", "sigmoid"]
    window_types = ["fixed", "variable"]
    hyperparameters_naive = []
    hyperparameters_one_hot = []
    max_f_score_naive = - sys.maxsize
    max_f_score_one_hot = - sys.maxsize
    for distance_type in distance_types:
        for kernel_type in kernel_types:
            for window_type in window_types:
                for h_k in range(1, h_k_range):
                    confusion_matrix_naive = np.zeros((K, K), dtype=int)
                    confusion_matrix_one_hot = np.zeros((K, K), dtype=int)
                    for i in range(len(dataset_naive)):
                        real_naive = int(dataset_naive[i][13]) - 1
                        real_one_hot = int(np.argmax(dataset_one_hot[i][13:]))
                        predict_naive = round(
                            nr.knnClassifyNaive(len(dataset_naive) - 1, F, dataset_naive[0:i] + dataset_naive[i + 1:],
                                                dataset_naive[i][0:F], distance_type, kernel_type, window_type,
                                                h_k)) - 1
                        predict_one_hot = int(np.argmax(nr.knnClassifyOneHot(len(dataset_one_hot) - 1, F, K,
                                                                             dataset_one_hot[0:i] + dataset_one_hot[
                                                                                                    i + 1:],
                                                                             dataset_one_hot[i][0:F], distance_type,
                                                                             kernel_type, window_type, h_k)))
                        confusion_matrix_naive[real_naive][predict_naive] += 1
                        confusion_matrix_one_hot[real_one_hot][predict_one_hot] += 1
                    f_score_naive = fs.getFscore(K, confusion_matrix_naive)
                    f_score_one_hot = fs.getFscore(K, confusion_matrix_one_hot)
                    if f_score_naive > max_f_score_naive:
                        hyperparameters_naive = [distance_type, kernel_type, window_type, h_k]
                        max_f_score_naive = f_score_naive
                    if f_score_one_hot > max_f_score_one_hot:
                        hyperparameters_one_hot = [distance_type, kernel_type, window_type, h_k]
                        max_f_score_one_hot = f_score_naive
    print(max_f_score_naive, hyperparameters_naive)
    print(max_f_score_one_hot, hyperparameters_one_hot)
    return (max_f_score_naive, hyperparameters_naive, max_f_score_one_hot, hyperparameters_one_hot)


def getPoints(dataset_naive, dataset_one_hot, K, F, h_k_range, hyperparameters_naive, hyperparameters_one_hot):
    naive_x = []
    naive_y = []
    one_hot_x = []
    one_hot_y = []
    for h_k in range(1, h_k_range):
        confusion_matrix_naive = np.zeros((K, K), dtype=int)
        confusion_matrix_one_hot = np.zeros((K, K), dtype=int)
        for i in range(len(dataset_naive)):
            real_naive = int(dataset_naive[i][13]) - 1
            real_one_hot = int(np.argmax(dataset_one_hot[i][13:]))
            predict_naive = round(
                nr.knnClassifyNaive(len(dataset_naive) - 1, F, dataset_naive[0:i] + dataset_naive[i + 1:],
                                    dataset_naive[i][0:F], hyperparameters_naive[0], hyperparameters_naive[1], hyperparameters_naive[3],
                                    h_k)) - 1
            predict_one_hot = int(np.argmax(nr.knnClassifyOneHot(len(dataset_one_hot) - 1, F, K,
                                                                 dataset_one_hot[0:i] + dataset_one_hot[
                                                                                        i + 1:],
                                                                 dataset_one_hot[i][0:F], hyperparameters_one_hot[0],
                                                                 hyperparameters_one_hot[1], hyperparameters_one_hot[2], h_k)))
            confusion_matrix_naive[real_naive][predict_naive] += 1
            confusion_matrix_one_hot[real_one_hot][predict_one_hot] += 1
        f_score_naive = fs.getFscore(K, confusion_matrix_naive)
        f_score_one_hot = fs.getFscore(K, confusion_matrix_one_hot)
        naive_x.append(h_k)
        naive_y.append(f_score_naive)
        one_hot_x.append(h_k)
        one_hot_y.append(f_score_one_hot)

    return [naive_x, naive_y, one_hot_x, one_hot_y]


# get dataset from csv

filename = "dataset_191_wine.csv"

dataset = pd.read_csv(filename)
dataset = dataset[
    ["Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols", "Flavanoids",
     "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280%2FOD315_of_diluted_wines", "Proline",
     "class"]]

# normalize it

minmax = minmax(dataset.values)
normalized_dataset_values = normalize(dataset.values, minmax)

# Reducing to a regression problem

# 1. Naive way. We already have classes of wine as 1,2 and 3.

naive_values = normalized_dataset_values.tolist()

# 2. One-Hot encoding
one_hot_values = normalized_dataset_values.tolist()
for i in range(len(one_hot_values)):
    class_ = one_hot_values[i][13]
    if one_hot_values[i][13] == 1.0:
        del one_hot_values[i][-1]
        one_hot_values[i].append(1)
        one_hot_values[i].append(0)
        one_hot_values[i].append(0)
    elif one_hot_values[i][13] == 2.0:
        del one_hot_values[i][-1]
        one_hot_values[i].append(0)
        one_hot_values[i].append(1)
        one_hot_values[i].append(0)
    elif one_hot_values[i][13] == 3.0:
        del one_hot_values[i][-1]
        one_hot_values[i].append(0)
        one_hot_values[i].append(0)
        one_hot_values[i].append(1)

# choose best hyperparameters

best_hyperparameters = getBestHyperparameters(naive_values, one_hot_values, 3, 13, 20)

#  get points
all_points = getPoints(naive_values, one_hot_values, 3, 13, 20, best_hyperparameters[1], best_hyperparameters[3])

# draw plot for naive
if best_hyperparameters[1][2] == "fixed":
    x_title = 'h'
else:
    x_title = 'k'
d = {x_title: np.array(all_points[0]), 'F-score': np.array(all_points[1])}
df = pd.DataFrame(d)
sb.lineplot(x=x_title, y="F-score", data=df)
plt.show()

#draw plot for one-hot


if best_hyperparameters[3][2] == "fixed":
    x_title = 'h'
else:
    x_title = 'k'
d = {x_title: np.array(all_points[2]), 'F-score': np.array(all_points[3])}
df = pd.DataFrame(d)
sb.lineplot(x=x_title, y="F-score", data=df)
plt.show()

