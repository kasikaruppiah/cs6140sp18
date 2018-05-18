from __future__ import division
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import numpy as np


class PerceptronKernel:
    def __init__(self, dataSet, epochs=100, linear=True, kFold=10):
        self.dataSet = dataSet
        self.epochs = epochs
        self.kFold = kFold
        self.kf = KFold(n_splits=kFold, shuffle=True)
        self.linear = linear

    def zScoreNormalization(self, dataSet, attributeMeans=[],
                            attributeStds=[]):
        normalizedDataSet = dataSet.copy(True)
        attributes = dataSet.shape[1] - 1

        if len(attributeMeans) == attributes:
            for i in range(attributes):
                attributeMean = attributeMeans[i]
                attributeStd = attributeStds[i]
                normalizedDataSet[i] = normalizedDataSet[i].apply(
                    lambda x: (x - attributeMean) / attributeStd if attributeStd > 0 else 0
                )

            return normalizedDataSet, attributeMeans, attributeStds
        else:
            newAttributeMeans = []
            newAttributeStds = []

            for i in range(attributes):
                attributeMean = dataSet[i].mean()
                newAttributeMeans.append(attributeMean)
                attributeStd = dataSet[i].std()
                newAttributeStds.append(attributeStd)
                normalizedDataSet[i] = normalizedDataSet[i].apply(
                    lambda x: (x - attributeMean) / attributeStd if attributeStd > 0 else 0
                )

            return normalizedDataSet, newAttributeMeans, newAttributeStds

    def constantFeature(self, dataSet):
        regressionDataSet = dataSet.copy(True)

        regressionDataSet.columns = range(1, regressionDataSet.shape[1] + 1)
        regressionDataSet.insert(0, 0, 1)

        return regressionDataSet

    def linearKernel(self, xJ, xI):
        return np.dot(xJ, xI.T)

    def rbfKernel(self, xJ, xI, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(xJ - xI)**2)

    def fit(self, dataSet):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        weights = np.matrix(np.zeros(dataSet.shape[0])).T
        epochs = 0

        kernels = [[self.linearKernel(X[j], X[i]) for j in range(X.shape[0])]
                   for i in range(X.shape[0])] if self.linear else [[
                       self.rbfKernel(X[j], X[i]) for j in range(X.shape[0])
                   ] for i in range(X.shape[0])]

        for _ in range(self.epochs):
            epochs = _ + 1
            loopError = 0
            for i in range(X.shape[0]):
                sumVal = 0
                for j in range(X.shape[0]):
                    sumVal += weights[j] * Y[j] * kernels[i][j]
                predictedVal = 1 if sumVal >= 0.0 else -1
                if Y[i] != predictedVal:
                    weights[i] += 1
                    loopError += 1
            if loopError == 0:
                break

        return epochs, weights

    def accuracy(self, trainDataSet, testDataSet, weights):
        trainX = np.matrix(trainDataSet.iloc[:, :-1])
        trainY = np.matrix(trainDataSet.iloc[:, -1]).T
        testX = np.matrix(testDataSet.iloc[:, :-1])
        testY = np.matrix(testDataSet.iloc[:, -1]).T

        prediction = []
        for i in range(testDataSet.shape[0]):
            sumVal = 0
            for j in range(trainDataSet.shape[0]):
                kernel = self.linearKernel(
                    trainX[j], testX[i]) if self.linear else self.rbfKernel(
                        trainX[j], testX[i])
                sumVal += weights[j] * trainY[j] * kernel
            predictedVal = 1 if sumVal >= 0.0 else -1
            prediction.append(predictedVal)

        accuracy = accuracy_score(testY, prediction)

        return accuracy

    def validate(self):
        epochs = []
        accuracies = []

        fold = 1

        print("Fold\tEpochs\tAccuracy")
        for trainIndex, testIndex in self.kf.split(self.dataSet):

            # trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
            #     )
            trainDataSet = self.constantFeature(self.dataSet.iloc[trainIndex])

            # testDataSet, _, _ = self.zScoreNormalization(
            #     , trainAttributeMeans,
            #     trainAttributeStds)
            testDataSet = self.constantFeature(self.dataSet.iloc[testIndex])

            epoch, weights = self.fit(trainDataSet)
            epochs.append(epoch)

            accuracy = self.accuracy(trainDataSet, testDataSet, weights)
            accuracies.append(accuracy)

            print("{}\t{}\t{}".format(fold, epoch, accuracy))
            fold += 1

        print("Mean\t{}\t{}".format(np.mean(epochs), np.mean(accuracies)))

        print("Standard Deviation\t{}\t{}".format(
            np.std(epochs), np.std(accuracies)))