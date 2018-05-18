from __future__ import division
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import zscore

import matplotlib.pyplot as plt


def Entropy_ClassLabel(y):
    dict_labelcount = {}
    y = list(y)
    for label in y:
        if label in dict_labelcount:
            dict_labelcount[label] += 1
        else:
            dict_labelcount[label] = 1
    total_count_labels = len(y)

    H_Y = 0
    for label in dict_labelcount:
        labelcount = dict_labelcount[label]
        l_ratio = (labelcount / total_count_labels)
        H_Y += (-1.0) * l_ratio * (np.math.log(l_ratio, 2))
    return H_Y


def GuassianNormal(X, c, s):
    term1 = 1 / (((2 * np.pi)**(len(c) / 2)) * (np.linalg.det(s)**(1 / 2)))
    term2 = np.exp(-0.5 * np.einsum('ij, ij -> i', X - c,
                                    np.dot(np.linalg.inv(s), (X - c).T).T))

    return term1 * term2


dataset = pd.read_csv('ecoliData.csv')

dataset = dataset.fillna(0)
dataset.iloc[:, :-1] = dataset.iloc[:, :-1].apply(zscore)
dataset = dataset.fillna(0)
dataset = dataset.drop_duplicates()
dataset = shuffle(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

H_Y = Entropy_ClassLabel(y)

list_clusters = range(1, 11)
list_SSE = []
list_NMI = []
best_SSE = float("inf")
tolerance = 0.00001

for k_means in list_clusters:

    max_iters = 100

    no_inputs, no_features = X.shape

    centroids = X[np.random.choice(len(X), k_means, False), :]

    covariance = [np.eye(X.shape[1])] * k_means

    probabilities_cluster = [1 / k_means] * k_means

    responsibility = np.zeros((no_inputs, k_means))
    log_likelihoods = []

    classifications = {}
    label_classifications = {}
    for k in range(k_means):
        classifications[k] = []
        label_classifications[k] = []

    iteration = 0

    P = lambda mu, s: (1/ ((np.linalg.det(s) ** (1/2)) * ((2 * np.pi) ** (len(mu) / 2)))) * np.exp(-.5 * np.einsum('ij, ij -> i', X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

    while len(log_likelihoods) < max_iters:

        for k in range(k_means):
            # responsibility[:, k] = probabilities_cluster[k] * P(centroids[k], covariance[k])
            responsibility[:, k] = probabilities_cluster[k] * GuassianNormal(
                X, centroids[k], covariance[k])
            print(k)

        log_likelihood = np.sum(np.log(np.sum(responsibility, axis=1)))
        log_likelihoods.append(log_likelihood)

        responsibility = (responsibility.T / np.sum(responsibility, axis=1)).T
        points_perCluster = np.sum(responsibility, axis=0)

        for k in range(k_means):
            centroids[k] = 1. / points_perCluster[k] * np.sum(
                responsibility[:, k] * X.T, axis=1).T
            x_mu = np.matrix(X - centroids[k])

            covariance[k] = np.array(1 / points_perCluster[k] * np.dot(
                np.multiply(x_mu.T, responsibility[:, k]), x_mu))

            probabilities_cluster[k] = 1. / no_inputs * points_perCluster[k]
        if len(log_likelihoods) > 2:
            if np.abs(log_likelihood - log_likelihoods[-2]) < tolerance:
                break

    for j in range(len(X)):
        classification = np.argmax(responsibility[j])
        classifications[classification].append(X[j])
        label_classifications[classification].append(y[j])

    squared_error = 0
    for k in range(k_means):
        kth_cluster = classifications[k]
        kth_centroid = centroids[k]
        for i in range(len(kth_cluster)):
            error = (sum((kth_centroid - kth_cluster[i])**2))
            squared_error += error
    print('The squared error with ', k_means, ' clusters is : ', squared_error)
    list_SSE.append(squared_error)
    if squared_error < best_SSE:
        best_SSE = squared_error

    total_labels = len(y)
    H_C = 0
    for classification in label_classifications:
        l_perclass = len(label_classifications[classification])
        l_perclass_ratio = l_perclass / total_labels
        H_C += (-1.0) * (np.math.log(l_perclass_ratio, 2)) * l_perclass_ratio
    # NMI calculation
    H_YC = 0
    for classification in label_classifications:
        cluster = label_classifications[classification]

        l_perclass = len(cluster)
        p = l_perclass / total_labels
        H_YC += p * Entropy_ClassLabel(cluster)

    I_YC = H_Y - H_YC
    NMI_Score = (2.0 * I_YC / (H_Y + H_C))
    print('The NMI score is : ', NMI_Score, " for the ", k_means,
          ' clusters !!')
    list_NMI.append(NMI_Score)

print(best_SSE)
plt.suptitle("Determining Best-K means using Elbow Method")
plt.plot(list_clusters, list_SSE, 'bo-')
plt.ylabel("SSE")
plt.xlabel("K value ")
plt.show()
