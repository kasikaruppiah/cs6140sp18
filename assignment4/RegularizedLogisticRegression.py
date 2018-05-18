from __future__ import division
from sklearn.model_selection import KFold

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RegularizedLogisticRegression:
    def __init__(self,
                 dataSet,
                 filename,
                 learningRate=0.001,
                 tolerance=0.001,
                 beta=0.05,
                 maxIterations=1000,
                 kFold=10):
        self.dataSet = dataSet
        self.filename = filename
        self.learningRate = learningRate
        self.tolerance = tolerance
        self.beta = beta
        self.maxIterations = maxIterations
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

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def hypothesis(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def cost(self, X, Y, theta):
        probabilities = self.hypothesis(X, theta)

        return (np.multiply(-Y, np.log(probabilities)) - np.multiply(
            (1 - Y), np.log(1 - probabilities))).mean()

    def gradient(self, X, Y, theta):
        prediction = self.hypothesis(X, theta)
        error = prediction - Y

        return np.dot(X.T, error) / X.shape[0]

    def logisticRegression(self, dataSet, plotGraph):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        theta = np.matrix(np.zeros(dataSet.shape[1] - 1)).T

        logisticLoss = [self.cost(X, Y, theta)]
        iterations = 0
        for i in range(self.maxIterations):
            iterations = i + 1
            gradient = self.gradient(X, Y, theta)
            newTheta = theta - self.learningRate * gradient
            newCost = self.cost(X, Y, newTheta)
            if (logisticLoss[i] - newCost < self.tolerance):
                break
            else:
                logisticLoss.append(newCost)
                theta = newTheta

        if plotGraph:
            plt.plot(logisticLoss, label='Logistic Regression')

        return theta, iterations

    def regCost(self, X, Y, theta):
        probabilities = self.hypothesis(X, theta)
        tempTheta = theta.copy()
        tempTheta[0] = 0

        cost = (np.multiply(-Y, np.log(probabilities)) - np.multiply(
            (1 - Y), np.log(1 - probabilities))).mean() + (self.beta / (
                2 * X.shape[0])) * np.sum(np.square(tempTheta))

        return cost

    def regGradient(self, X, Y, theta):
        prediction = self.hypothesis(X, theta)
        error = prediction - Y

        return np.dot(X.T, error) * self.beta / X.shape[0]

    def regLogisticRegression(self, dataSet, plotGraph):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        theta = np.matrix(np.zeros(dataSet.shape[1] - 1)).T

        logisticLoss = [self.regCost(X, Y, theta)]
        iterations = 0
        for i in range(self.maxIterations):
            iterations = i + 1
            gradient = self.regGradient(X, Y, theta)
            tempTheta = theta.copy()
            tempTheta[0] = 0
            newTheta = theta - self.learningRate * gradient - (
                self.beta / X.shape[0]) * tempTheta
            newCost = self.regCost(X, Y, newTheta)
            if (logisticLoss[i] - newCost < self.tolerance):
                break
            else:
                logisticLoss.append(newCost)
                theta = newTheta

        if plotGraph:
            plt.plot(logisticLoss, label='Regularized Logistic Regression')

        return theta, iterations

    def predict(self, X, theta):
        probabilities = self.hypothesis(X, theta)

        return [
            1 if probability >= 0.5 else 0 for probability in probabilities
        ]

    def accuracy(self, dataSet, theta):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = dataSet.iloc[:, -1]

        prediction = self.predict(X, theta)

        accuracy = (prediction == Y).mean()

        Y = Y.values.tolist()
        truePositive = 0
        for index, value in enumerate(prediction):
            if value == 1 and Y[index] == 1:
                truePositive += 1

        precision = truePositive / prediction.count(1)
        recall = truePositive / Y.count(1)

        return accuracy, precision, recall

    def validate(self):
        testIterations = []
        testAccuracies = []
        testPrecisions = []
        testRecalls = []
        regTestIterations = []
        regTestAccuracies = []
        regTestPrecisions = []
        regTestRecalls = []

        fold = 1
        plotFold = random.randint(1, self.kFold + 1)
        plt.figure()
        print(
            "Fold\tIterations\tTest Accuracy\tTest Precision\tTest Recall\tReg Iterations\tReg Test Accuracy\tReg Test Precision\tReg Test Recall"
        )
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
                self.dataSet.iloc[trainIndex])
            trainDataSet = self.constantFeature(trainDataSet)
            testDataSet, _, _ = self.zScoreNormalization(
                self.dataSet.iloc[testIndex], trainAttributeMeans,
                trainAttributeStds)
            testDataSet = self.constantFeature(testDataSet)

            theta, iterations = self.logisticRegression(
                trainDataSet, fold == plotFold)
            testIterations.append(iterations)

            testAccuracy, testPrecision, testRecall = self.accuracy(
                testDataSet, theta)
            testAccuracies.append(testAccuracy)
            testPrecisions.append(testPrecision)
            testRecalls.append(testRecall)

            regTheta, regIterations = self.regLogisticRegression(
                trainDataSet, fold == plotFold)
            regTestIterations.append(regIterations)

            regTestAccuracy, regTestPrecision, regTestRecall = self.accuracy(
                testDataSet, regTheta)
            regTestAccuracies.append(regTestAccuracy)
            regTestPrecisions.append(regTestPrecision)
            regTestRecalls.append(regTestRecall)

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                fold, iterations, testAccuracy, testPrecision, testRecall,
                regIterations, regTestAccuracy, regTestPrecision,
                regTestRecall))
            fold += 1

        print("Mean\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.mean(testIterations), np.mean(testAccuracies),
            np.mean(testPrecisions), np.mean(testRecalls),
            np.mean(regTestIterations), np.mean(regTestAccuracies),
            np.mean(regTestPrecisions), np.mean(regTestRecalls)))
        print("Standard Deviation\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.std(testIterations), np.std(testAccuracies),
            np.std(testPrecisions), np.std(testRecalls),
            np.std(regTestIterations), np.std(regTestAccuracies),
            np.std(regTestPrecisions), np.std(regTestRecalls)))

        plt.xlabel('Iteration')
        plt.ylabel('Logistic Loss')
        plt.title('Regularized Logistic Regression')
        plt.savefig('Regression_Comparison_{}'.format(self.filename))
