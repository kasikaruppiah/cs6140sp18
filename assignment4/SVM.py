from __future__ import division
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np


class SVM:
    def __init__(self, dataSet, linear=True, kFold=10, mFold=5):
        self.dataSet = dataSet.fillna(0)
        self.linear = linear

        self.kFold = kFold
        self.kf = KFold(n_splits=kFold, shuffle=True)

        self.mFold = mFold
        self.mf = KFold(n_splits=mFold, shuffle=True)

        self.cList = [2**i for i in range(-5, 11)]
        self.gammaList = [2**i for i in range(-15, 6)]

    def linearSVC(self):
        bestCs = []
        trainAccuracies = []
        trainPrecisions = []
        trainRecalls = []
        accuracies = []
        precisions = []
        recalls = []

        fold = 1

        print(
            "Fold\tC\tTrain Accuracy\tTrain Precision\tTrain Recall\tTest Accuracy\tTest Precision\tTest Recall"
        )
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet = self.dataSet.iloc[trainIndex]
            xTrain = trainDataSet.iloc[:, :-1]
            yTrain = trainDataSet.iloc[:, -1].T

            nestedTrainDataSets = []
            nestedTestDataSets = []
            for nestedTrainIndex, nestedTestIndex in self.mf.split(
                    trainDataSet):
                nestedTrainDataSets.append(trainDataSet.iloc[nestedTrainIndex])
                nestedTestDataSets.append(trainDataSet.iloc[nestedTestIndex])

            highestAccuracy = -1
            bestC = 0.0

            for nestedC in self.cList:
                nestedAccuracy = 0.0
                for i in range(self.mFold):
                    nestedTrainDataSet = nestedTrainDataSets[i]
                    xNestedTrain = nestedTrainDataSet.iloc[:, :-1]
                    yNestedTrain = nestedTrainDataSet.iloc[:, -1].T

                    nestedClf = SVC(kernel='linear', C=nestedC)
                    nestedClf.fit(xNestedTrain, yNestedTrain)

                    nestedTestDataSet = nestedTestDataSets[i]
                    xNestedTest = nestedTestDataSet.iloc[:, :-1]
                    yNestedTest = nestedTestDataSet.iloc[:, -1].T

                    yNestedPredicted = nestedClf.predict(xNestedTest)
                    nestedAccuracy += accuracy_score(yNestedTest,
                                                     yNestedPredicted)

                nestedAccuracy /= self.mFold
                if (nestedAccuracy >= highestAccuracy):
                    highestAccuracy = nestedAccuracy
                    bestC = nestedC

            bestCs.append(bestC)

            clf = SVC(kernel='linear', C=bestC)
            clf.fit(xTrain, yTrain)
            yTrainPredicted = clf.predict(xTrain)
            trainAccuracy = accuracy_score(yTrain, yTrainPredicted)
            trainPrecision = precision_score(yTrain, yTrainPredicted)
            trainRecall = recall_score(yTrain, yTrainPredicted)

            trainAccuracies.append(trainAccuracy)
            trainPrecisions.append(trainPrecision)
            trainRecalls.append(trainRecall)

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, :-1]
            yTest = testDataSet.iloc[:, -1].T

            yPredicted = clf.predict(xTest)
            accuracy = accuracy_score(yTest, yPredicted)
            precision = precision_score(yTest, yPredicted)
            recall = recall_score(yTest, yPredicted)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                fold, bestC, trainAccuracy, trainPrecision, trainRecall,
                accuracy, precision, recall))

            fold += 1

        print("Mean\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.mean(bestCs),
            np.mean(trainAccuracies), np.mean(trainPrecisions),
            np.mean(trainRecalls), np.mean(accuracies), np.mean(precisions),
            np.mean(recalls)))
        print("Standard Deviation\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.std(bestCs), np.std(trainAccuracies), np.std(trainPrecisions),
            np.std(trainRecalls), np.std(accuracies), np.std(precisions),
            np.std(recalls)))

    def rbfSVC(self):
        bestCs = []
        bestGammas = []
        trainAccuracies = []
        trainPrecisions = []
        trainRecalls = []
        accuracies = []
        precisions = []
        recalls = []

        fold = 1

        print(
            "Fold\tC\tGamma\tTrain Accuracy\tTrain Precision\tTrain Recall\tTest Accuracy\tTest Precision\tTest Recall"
        )
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet = self.dataSet.iloc[trainIndex]
            xTrain = trainDataSet.iloc[:, :-1]
            yTrain = trainDataSet.iloc[:, -1].T

            nestedTrainDataSets = []
            nestedTestDataSets = []
            for nestedTrainIndex, nestedTestIndex in self.mf.split(
                    trainDataSet):
                nestedTrainDataSets.append(trainDataSet.iloc[nestedTrainIndex])
                nestedTestDataSets.append(trainDataSet.iloc[nestedTestIndex])

            highestAccuracy = -1
            bestC = 0.0
            bestGamma = 0.0

            for nestedC in self.cList:
                for nestedGamma in self.gammaList:
                    nestedAccuracy = 0.0
                    for i in range(self.mFold):
                        nestedTrainDataSet = nestedTrainDataSets[i]
                        xNestedTrain = nestedTrainDataSet.iloc[:, :-1]
                        yNestedTrain = nestedTrainDataSet.iloc[:, -1].T

                        nestedClf = SVC(C=nestedC, gamma=nestedGamma)
                        nestedClf.fit(xNestedTrain, yNestedTrain)

                        nestedTestDataSet = nestedTestDataSets[i]
                        xNestedTest = nestedTestDataSet.iloc[:, :-1]
                        yNestedTest = nestedTestDataSet.iloc[:, -1].T

                        yNestedPredicted = nestedClf.predict(xNestedTest)
                        nestedAccuracy += accuracy_score(
                            yNestedTest, yNestedPredicted)

                    nestedAccuracy /= self.mFold
                    if (nestedAccuracy >= highestAccuracy):
                        highestAccuracy = nestedAccuracy
                        bestC = nestedC
                        bestGamma = nestedGamma

            bestCs.append(bestC)
            bestGammas.append(bestGamma)

            clf = SVC(C=bestC, gamma=bestGamma)
            clf.fit(xTrain, yTrain)
            yTrainPredicted = clf.predict(xTrain)
            trainAccuracy = accuracy_score(yTrain, yTrainPredicted)
            trainPrecision = precision_score(yTrain, yTrainPredicted)
            trainRecall = recall_score(yTrain, yTrainPredicted)

            trainAccuracies.append(trainAccuracy)
            trainPrecisions.append(trainPrecision)
            trainRecalls.append(trainRecall)

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, :-1]
            yTest = testDataSet.iloc[:, -1].T

            yPredicted = clf.predict(xTest)
            accuracy = accuracy_score(yTest, yPredicted)
            precision = precision_score(yTest, yPredicted)
            recall = recall_score(yTest, yPredicted)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                fold, bestC, bestGamma, trainAccuracy, trainPrecision,
                trainRecall, accuracy, precision, recall))

            fold += 1

        print("Mean\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.mean(bestCs), np.mean(bestGammas), np.mean(trainAccuracies),
            np.mean(trainPrecisions), np.mean(trainRecalls),
            np.mean(accuracies), np.mean(precisions), np.mean(recalls)))
        print("Standard Deviation\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.std(bestCs), np.std(bestGammas), np.std(trainAccuracies),
            np.std(trainPrecisions), np.std(trainRecalls), np.std(accuracies),
            np.std(precisions), np.std(recalls)))

    def validate(self):
        if self.linear:
            self.linearSVC()
        else:
            self.rbfSVC()
