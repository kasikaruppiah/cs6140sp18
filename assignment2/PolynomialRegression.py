from __future__ import division
from sklearn.model_selection import KFold
from numpy.linalg import inv

import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt


class PolynomialRegression:
    def __init__(self,
                 trainingDataSet,
                 validationDataSet,
                 kFold,
                 p,
                 normalizeDataSet=True):
        self.trainingDataSet = trainingDataSet.fillna(0)
        self.validationDataSet = validationDataSet.fillna(
            0) if validationDataSet is not None else None
        self.kf = KFold(n_splits=kFold, shuffle=True) if kFold else None
        self.p = p
        self.normalizeDataSet = normalizeDataSet

    def polynomialDataSet(self, dataSet, p):
        columns = dataSet.shape[1] - 1
        newDataSet = pd.DataFrame()

        for i in range(1, p + 1):
            newDataSet[range(columns * (i - 1), columns * i)] = numpy.power(
                dataSet[range(columns)], i)
        newDataSet[columns * p] = dataSet[columns]

        return newDataSet

    def zScoreNormalization(self,
                            dataSet,
                            p,
                            attributeMeans=[],
                            attributeStds=[]):
        normalizedDataSet = self.polynomialDataSet(dataSet, p)
        if self.normalizeDataSet:
            attributes = normalizedDataSet.shape[1] - 1

            if len(attributeMeans) == attributes:
                for i in range(attributes):
                    attributeMean = attributeMeans[i]
                    attributeStd = attributeStds[i]
                    normalizedDataSet[i] = normalizedDataSet[i].apply(
                        lambda x: (x - attributeMean) / attributeStd if attributeStd > 0 else 0
                    )
            else:
                newAttributeMeans = []
                newAttributeStds = []

                for i in range(attributes):
                    attributeMean = normalizedDataSet[i].mean()
                    newAttributeMeans.append(attributeMean)
                    attributeStd = normalizedDataSet[i].std()
                    newAttributeStds.append(attributeStd)
                    normalizedDataSet[i] = normalizedDataSet[i].apply(
                        lambda x: (x - attributeMean) / attributeStd if attributeStd > 0 else 0
                    )

                return normalizedDataSet, newAttributeMeans, newAttributeStds

        return normalizedDataSet, attributeMeans, attributeStds

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
        pTrainRMSE = []
        pTestRMSE = []
        pTrainSSE = []
        pTestSSE = []
        for p in self.p:
            print("p :: {}".format(p))
            if self.validationDataSet is not None:
                print("Training SSE\tTraining RMSE\tTest SSE\tTest RMSE")
                trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
                    self.trainingDataSet, p)
                trainDataSet = self.constantFeature(trainDataSet)
                w = self.normalEquation(trainDataSet)
                trainSumSquaredErrors, trainRootMeanSquareError = self.rootMeanSquareError(
                    trainDataSet, w)
                pTrainSSE.append(trainSumSquaredErrors / trainDataSet.shape[0])
                testDataSet, testAttributeMeans, testAttributeStds = self.zScoreNormalization(
                    self.validationDataSet, p, trainAttributeMeans,
                    trainAttributeStds)
                testDataSet = self.constantFeature(testDataSet)
                testSumSquaredErrors, testRootMeanSquareError = self.rootMeanSquareError(
                    testDataSet, w)
                pTestSSE.append(testSumSquaredErrors / testDataSet.shape[0])
                print("{}\t{}\t{}\t{}".format(
                    trainSumSquaredErrors, trainRootMeanSquareError,
                    testSumSquaredErrors, testRootMeanSquareError))
            else:
                trainSSE = []
                trainRMSE = []
                testSSE = []
                testRMSE = []
                fold = 1
                print("Fold\tTraining SSE\tTraining RMSE\tTest SSE\tTest RMSE")
                for trainIndex, testIndex in self.kf.split(
                        self.trainingDataSet):
                    trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
                        self.trainingDataSet.iloc[trainIndex], p)
                    trainDataSet = self.constantFeature(trainDataSet)
                    w = self.normalEquation(trainDataSet)
                    trainSumSquaredErrors, trainRootMeanSquareError = self.rootMeanSquareError(
                        trainDataSet, w)
                    trainSSE.append(trainSumSquaredErrors)
                    trainRMSE.append(trainRootMeanSquareError)
                    testDataSet, testAttributeMeans, testAttributeStds = self.zScoreNormalization(
                        self.trainingDataSet.iloc[testIndex], p,
                        trainAttributeMeans, trainAttributeStds)
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
                pTrainRMSE.append(numpy.mean(trainRMSE))
                pTestRMSE.append(numpy.mean(testRMSE))
                print("{}\t{}\t{}\t{}\t{}".format('Standard Deviation',
                                                  numpy.std(trainSSE),
                                                  numpy.std(trainRMSE),
                                                  numpy.std(testSSE),
                                                  numpy.std(testRMSE)))
        if self.validationDataSet is not None:
            plt.plot(self.p, pTrainSSE, label='Training Data Set')
            plt.plot(self.p, pTestSSE, label='Test Data Set')
            plt.xlabel('p')
            plt.ylabel('Mean SSE')
            plt.title('Polynomial Regression')
            plt.legend(loc='best')
            plt.show()
        else:
            plt.plot(self.p, pTrainRMSE, label='Training Data Set')
            plt.plot(self.p, pTestRMSE, label='Test Data Set')
            plt.xlabel('p')
            plt.ylabel('Mean RMSE')
            plt.title('Polynomial Regression')
            plt.legend(loc='best')
            plt.show()
