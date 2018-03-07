from __future__ import division
from sklearn.model_selection import KFold
from numpy.linalg import inv

import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt


class RidgeRegression:
    def __init__(self, dataSet, kFold, p, c):
        self.dataSet = dataSet.fillna(0)
        self.kf = KFold(n_splits=kFold, shuffle=True)
        self.p = p
        self.c = c

    def polynomialDataSet(self, dataSet, p):
        columns = dataSet.shape[1] - 1
        newDataSet = pd.DataFrame()

        for i in range(1, p + 1):
            newDataSet[range(columns * (i - 1), columns * i)] = numpy.power(
                dataSet[range(columns)], i)
        newDataSet[columns * p] = dataSet[columns]

        return newDataSet

    def zScoreNormalization(self, dataSet, p, attributeMeans=[]):
        normalizedDataSet = self.polynomialDataSet(dataSet, p)
        attributes = normalizedDataSet.shape[1] - 1

        if len(attributeMeans) == attributes:
            for i in range(attributes):
                attributeMean = attributeMeans[i]
                normalizedDataSet[i] = normalizedDataSet[i].apply(
                    lambda x: x - attributeMean)
            return normalizedDataSet, attributeMeans
        else:
            newAttributeMeans = []

            for i in range(attributes):
                attributeMean = normalizedDataSet[i].mean()
                newAttributeMeans.append(attributeMean)
                normalizedDataSet[i] = normalizedDataSet[i].apply(
                    lambda x: x - attributeMean)

            return normalizedDataSet, newAttributeMeans

    def ridgeRegression(self, dataSet, c):
        attributes = dataSet.shape[1] - 1
        X = dataSet.as_matrix(range(attributes))
        y = dataSet[attributes]

        w = numpy.dot(
            numpy.dot(
                inv(
                    numpy.dot(X.transpose(), X) +
                    numpy.dot(c, numpy.identity(attributes))), X.transpose()),
            y)
        w = numpy.insert(w, 0, y.mean())

        return w

    def predict(self, row, w):
        h = w[0]
        attributes = len(row) - 1

        for i in range(attributes):
            h += w[i + 1] * row[i]

        return h - row[attributes]

    def rootMeanSquareError(self, dataSet, w):
        sumSquaredErrors = 0.0

        for index, row in dataSet.iterrows():
            sumSquaredErrors += self.predict(row, w)**2

        return sumSquaredErrors, math.sqrt(sumSquaredErrors / dataSet.shape[0])

    def validate(self):
        for p in self.p:
            print("p :: {}".format(p))
            cTrainRMSE = []
            cTestRMSE = []
            for c in self.c:
                print("c :: {}".format(c))
                trainSSE = []
                trainRMSE = []
                testSSE = []
                testRMSE = []
                fold = 1
                print("Fold\tTraining SSE\tTraining RMSE\tTest SSE\tTest RMSE")
                for trainIndex, testIndex in self.kf.split(self.dataSet):
                    trainDataSet, trainAttributeMeans = self.zScoreNormalization(
                        self.dataSet.iloc[trainIndex], p)
                    w = self.ridgeRegression(trainDataSet, c)
                    trainSumSquaredErrors, trainRootMeanSquareError = self.rootMeanSquareError(
                        trainDataSet, w)
                    trainSSE.append(trainSumSquaredErrors)
                    trainRMSE.append(trainRootMeanSquareError)
                    testDataSet, testAttributeMeans = self.zScoreNormalization(
                        self.dataSet.iloc[testIndex],
                        p,
                        trainAttributeMeans,
                    )
                    testSumSquaredErrors, testRootMeanSquareError = self.rootMeanSquareError(
                        testDataSet, w)
                    testSSE.append(testSumSquaredErrors)
                    testRMSE.append(testRootMeanSquareError)
                    print("{}\t{}\t{}\t{}\t{}".format(
                        fold, trainSumSquaredErrors, trainRootMeanSquareError,
                        testSumSquaredErrors, testRootMeanSquareError))
                    fold += 1
                print("{}\t{}\t{}\t{}\t{}".format('Mean', numpy.mean(trainSSE),
                                                  numpy.mean(trainRMSE),
                                                  numpy.mean(testSSE),
                                                  numpy.mean(testRMSE)))
                cTrainRMSE.append(numpy.mean(trainRMSE))
                cTestRMSE.append(numpy.mean(testRMSE))
                print("{}\t{}\t{}\t{}\t{}".format('Standard Deviation',
                                                  numpy.std(trainSSE),
                                                  numpy.std(trainRMSE),
                                                  numpy.std(testSSE),
                                                  numpy.std(testRMSE)))
            plt.plot(self.c, cTrainRMSE, label='Training Data Set')
            plt.plot(self.c, cTestRMSE, label='Test Data Set')
            plt.xlabel('c')
            plt.ylabel('Mean RMSE')
            plt.title('Ridge Regression - {}'.format(p))
            plt.legend(loc='best')
            plt.show()
