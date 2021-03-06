from __future__ import division
from copy import deepcopy

import numpy as np
import collections
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class KMeansClustering:
    def __init__(self,
                 dataSet,
                 pltName,
                 K=range(1, 11),
                 tolerance=0.00001,
                 epochs=100):
        self.dataSet = dataSet
        self.pltName = pltName
        self.K = K
        self.tolerance = tolerance
        self.epochs = epochs

    def distance(self, a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

    def SSE(self, X, clusters, C):
        sse = 0.0
        for i in range(len(X)):
            sse += np.sum(np.square(np.subtract(X[i], C[int(clusters[i])])))
        return sse

    def NMI(self, Y, clusters, C):
        labelCounter = collections.Counter(Y)
        clusterCounter = collections.Counter(clusters)

        hy = 0.0
        for label in labelCounter:
            labelProb = labelCounter[label] / len(Y)
            if labelProb > 0.0:
                hy -= labelProb * math.log(labelProb, 2)

        cProbs = {}
        hc = 0.0
        for c in range(len(C)):
            cProb = clusterCounter[c] / len(clusters)
            cProbs[c] = cProb
            if cProb > 0.0:
                hc -= cProb * math.log(cProb, 2)

        hyc = 0.0
        for c in range(len(C)):
            classLabels = [Y[j] for j in range(len(Y)) if clusters[j] == c]
            classLabelCounter = collections.Counter(classLabels)
            nhy = 0.0
            for label in classLabelCounter:
                labelProb = classLabelCounter[label] / len(classLabels)
                if labelProb > 0.0:
                    nhy += labelProb * math.log(labelProb, 2)
            hyc -= cProbs[c] * nhy
        iyc = hy - hyc

        return 2 * iyc / (hy + hc)

    def validate(self):
        SSEs = []
        NMIs = []

        X = self.dataSet.iloc[:, :-1].values
        Y = self.dataSet.iloc[:, -1].values

        print("K\tSSE\tNMI")
        for k in self.K:
            C = X[np.random.choice(len(X), k, False), :]
            cOld = np.zeros(C.shape)
            clusters = np.zeros(len(X))

            error = self.tolerance + 1
            for _ in range(self.epochs):
                if error > self.tolerance:
                    for i in range(len(X)):
                        clusters[i] = np.argmin(self.distance(X[i], C))
                    cOld = deepcopy(C)
                    for i in range(k):
                        points = [
                            X[j] for j in range(len(X)) if clusters[j] == i
                        ]
                        if (len(points) > 0):
                            C[i] = np.mean(points, axis=0)
                    error = self.distance(C, cOld, None)
                else:
                    break
            sse = self.SSE(X, clusters, C)
            nmi = self.NMI(Y, clusters, C)

            print("{}\t{}\t{}".format(k, sse, nmi))

            SSEs.append(sse)
            NMIs.append(nmi)

        plt.figure()
        plt.plot(self.K, SSEs, 'ro-')
        plt.ylabel('SSE')
        plt.xlabel('K')
        plt.title("{} - K vs SSE".format(self.pltName))
        plt.savefig("K_Means_Clustering_SSE_{}".format(self.pltName))
