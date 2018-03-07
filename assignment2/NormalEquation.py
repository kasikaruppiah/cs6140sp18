from __future__ import division
from sklearn.model_selection import KFold
from numpy.linalg import inv

import numpy
import math


class NormalEquation:
    def __init__(self, dataSet, kFold):
        self.dataSet = dataSet.fillna(0)

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

    def normalEquation(self, dataSet):
        attributes = dataSet.shape[1] - 1
        X = dataSet.as_matrix(range(attributes))
        y = dataSet[attributes]

        w = numpy.dot(
            numpy.dot(inv(numpy.dot(X.transpose(), X)), X.transpose()), y)

        return w

    def predict(self, row, w):
        h = 0.0
        attributes = len(row) - 1

        for i in range(attributes):
            h += w[i] * row[i]

        return h - row[attributes]

    def rootMeanSquareError(self, dataSet, w):
        sumSquaredErrors = 0.0

        for index, row in dataSet.iterrows():
            sumSquaredErrors += self.predict(row, w)**2

        return sumSquaredErrors, math.sqrt(sumSquaredErrors / dataSet.shape[0])

    def validate(self):
        trainSSE = []
        trainRMSE = []
        testSSE = []
        testRMSE = []
        fold = 1
        print("Fold\tTraining SSE\tTraining RMSE\tTest SSE\tTest RMSE")
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
                self.dataSet.iloc[trainIndex])
            trainDataSet = self.constantFeature(trainDataSet)
            w = self.normalEquation(trainDataSet)
            trainSumSquaredErrors, trainRootMeanSquareError = self.rootMeanSquareError(
                trainDataSet, w)
            trainSSE.append(trainSumSquaredErrors)
            trainRMSE.append(trainRootMeanSquareError)
            testDataSet, testAttributeMeans, testAttributeStds = self.zScoreNormalization(
                self.dataSet.iloc[testIndex], trainAttributeMeans,
                trainAttributeStds)
            testDataSet = self.constantFeature(testDataSet)
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
        print("{}\t{}\t{}\t{}\t{}".format('Standard Deviation',
                                          numpy.std(trainSSE),
                                          numpy.std(trainRMSE),
                                          numpy.std(testSSE),
                                          numpy.std(testRMSE)))
