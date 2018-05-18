from __future__ import division

import numpy as np
import collections
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GMMClustering:
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

    def GuassianNormal(self, X, c, s):
        term1 = 1 / (((2 * np.pi)**(len(c) / 2)) * (np.linalg.det(s)**(1 / 2)))
        term2 = np.exp(-0.5 * np.einsum('ij, ij -> i', X - c,
                                        np.dot(np.linalg.inv(s), (X - c).T).T))

        return term1 * term2

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
            S = [np.eye(X.shape[1])] * k
            W = [1 / k] * k
            R = np.zeros((len(X), k))
            clusters = np.zeros(len(X))

            oldLogLikelihood = 0
            error = self.tolerance + 1
            for _ in range(self.epochs):
                if error > self.tolerance:
                    for nestedK in range(k):
                        R[:, nestedK] = W[nestedK] * self.GuassianNormal(
                            X, C[nestedK], S[nestedK])

                    R = (R.T / np.sum(R, axis=1)).T

                    dataPoints = np.sum(R, axis=0)

                    for nestedK in range(k):
                        C[nestedK] = 1 / dataPoints[nestedK] * np.sum(
                            R[:, nestedK] * X.T, axis=1).T
                        deltaC = np.matrix(X - C[nestedK])

                        S[nestedK] = np.array(1 / dataPoints[nestedK] * np.dot(
                            np.multiply(deltaC.T, R[:, nestedK]), deltaC))

                        W[nestedK] = 1 / len(X) * dataPoints[nestedK]

                    logLikelihood = np.sum(np.log(np.sum(R, axis=1)))

                    error = logLikelihood - oldLogLikelihood

                    oldLogLikelihood = logLikelihood
                else:
                    break

            for i in range(len(X)):
                clusters[i] = np.argmax(R[i])

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
        plt.savefig("GMM_Clustering_SSE_{}".format(self.pltName))

        plt.figure()
        plt.plot(self.K, NMIs, 'bo-')
        plt.ylabel('NMI')
        plt.xlabel('K')
        plt.title("{} - K vs NMI".format(self.pltName))
        plt.savefig("GMM_Clustering_NMI_{}".format(self.pltName))
