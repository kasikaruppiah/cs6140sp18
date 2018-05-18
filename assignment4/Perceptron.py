from __future__ import division
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import numpy as np


class Perceptron:
    def __init__(self, dataSet, learningRate=0.1, epochs=100, kFold=10):
        self.dataSet = dataSet
        self.learningRate = learningRate
        self.epochs = epochs
        self.kFold = kFold
        self.kf = KFold(n_splits=kFold, shuffle=True)

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

    def predict(self, x, weights):

        return 1 if np.dot(x, weights) >= 0.0 else -1

    def fit(self, dataSet):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        attributes = dataSet.shape[1] - 1
        weights = np.matrix(np.random.uniform(-1, 1, attributes)).T
        epochs = 0

        for _ in range(self.epochs):
            epochs = _ + 1
            loopError = 0
            for i in range(X.shape[0]):
                error = Y[i] - self.predict(X[i], weights)
                if error != 0:
                    weights += (self.learningRate * error * X[i]).T
                    loopError += 1
            if loopError == 0:
                break

        return epochs, weights

    def accuracy(self, dataSet, weights):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = dataSet.iloc[:, -1]

        prediction = []
        for i in range(dataSet.shape[0]):
            prediction.append(self.predict(X[i], weights))

        accuracy = accuracy_score(Y, prediction)

        return accuracy

    def linearKernel(self, xJ, xI):
        return np.dot(xJ, xI.T)

    def dualFit(self, dataSet):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        weights = np.matrix(np.zeros(dataSet.shape[0])).T
        epochs = 0

        linearKernels = [[
            self.linearKernel(X[j], X[i]) for j in range(X.shape[0])
        ] for i in range(X.shape[0])]

        for _ in range(self.epochs):
            epochs = _ + 1
            loopError = 0
            for i in range(X.shape[0]):
                sumVal = 0
                for j in range(X.shape[0]):
                    sumVal += weights[j] * Y[j] * linearKernels[i][j]
                predictedVal = 1 if sumVal >= 0.0 else -1
                if Y[i] != predictedVal:
                    weights[i] += 1
                    loopError += 1
            if loopError == 0:
                break

        return epochs, weights

    def dualAccuracy(self, trainDataSet, testDataSet, weights):
        trainX = np.matrix(trainDataSet.iloc[:, :-1])
        trainY = np.matrix(trainDataSet.iloc[:, -1]).T
        testX = np.matrix(testDataSet.iloc[:, :-1])
        testY = np.matrix(testDataSet.iloc[:, -1]).T

        prediction = []
        for i in range(testDataSet.shape[0]):
            sumVal = 0
            for j in range(trainDataSet.shape[0]):
                sumVal += weights[j] * trainY[j] * self.linearKernel(
                    trainX[j], testX[i])
            predictedVal = 1 if sumVal >= 0.0 else -1
            prediction.append(predictedVal)

        accuracy = accuracy_score(testY, prediction)

        return accuracy

    def validate(self):
        epochs = []
        accuracies = []
        dualEpochs = []
        dualAccuracies = []

        fold = 1

        print(
            "Fold\tEpochs\tPerceptron Accuracy\tDual Perceptron Epochs\tDual Perceptron Accuracy"
        )
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
                self.dataSet.iloc[trainIndex])
            trainDataSet = self.constantFeature(trainDataSet)

            testDataSet, _, _ = self.zScoreNormalization(
                self.dataSet.iloc[testIndex], trainAttributeMeans,
                trainAttributeStds)
            testDataSet = self.constantFeature(testDataSet)

            epoch, weights = self.fit(trainDataSet)
            epochs.append(epoch)

            accuracy = self.accuracy(testDataSet, weights)
            accuracies.append(accuracy)

            dualEpoch, dualWeights = self.dualFit(trainDataSet)
            dualEpochs.append(dualEpoch)

            dualAccuracy = self.dualAccuracy(trainDataSet, testDataSet,
                                             dualWeights)
            dualAccuracies.append(dualAccuracy)

            print("{}\t{}\t{}\t{}\t{}".format(fold, epoch, accuracy, dualEpoch,
                                              dualAccuracy))
            fold += 1

        print("Mean\t{}\t{}\t{}\t{}".format(
            np.mean(epochs), np.mean(accuracies), np.mean(dualEpochs),
            np.mean(dualAccuracies)))

        print("Standard Deviation\t{}\t{}\t{}\t{}".format(
            np.std(epochs), np.std(accuracies), np.std(dualEpochs),
            np.std(dualAccuracies)))
