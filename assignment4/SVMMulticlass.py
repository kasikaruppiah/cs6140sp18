from __future__ import division
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import numpy as np


class SVMMulticlass:
    def __init__(self, dataSet, linear=True, kFold=10, mFold=5):
        self.dataSet = dataSet
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

        print("Fold\tC1-C\tC2-C\tC3-C\
            \tC1-TrainAccuracy\tC2-TrainAccuracy\tC3-TrainAccuracy\
            \tC1-TrainPrecision\tC2-TrainPrecision\tC3-TrainPrecision\
            \tC1-TrainRecall\tC2-TrainRecall\tC3-TrainRecall\
            \tC1-TestAccuracy\tC2-TestAccuracy\tC3-TestAccuracy\
            \tC1-TestPrecision\tC2-TestPrecision\tC3-TestPrecision\
            \tC1-TestRecall\tC2-TestRecall\tC3-TestRecall")
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet = self.dataSet.iloc[trainIndex]
            xTrain = trainDataSet.iloc[:, 1:]
            yTrain = trainDataSet.iloc[:, 0].T

            nestedTrainDataSets = []
            nestedTestDataSets = []
            for nestedTrainIndex, nestedTestIndex in self.mf.split(
                    trainDataSet):
                nestedTrainDataSets.append(trainDataSet.iloc[nestedTrainIndex])
                nestedTestDataSets.append(trainDataSet.iloc[nestedTestIndex])

            highestAccuracy = [-1] * 3

            bestCs.append([0.0] * 3)
            for nestedC in self.cList:
                nestedAccuracy = 0.0
                for i in range(self.mFold):
                    nestedTrainDataSet = nestedTrainDataSets[i]
                    xNestedTrain = nestedTrainDataSet.iloc[:, 1:]
                    yNestedTrain = nestedTrainDataSet.iloc[:, 0].T

                    nestedClf = OneVsRestClassifier(
                        SVC(kernel='linear', C=nestedC))
                    nestedClf.fit(xNestedTrain, yNestedTrain)

                    nestedTestDataSet = nestedTestDataSets[i]
                    xNestedTest = nestedTestDataSet.iloc[:, 1:]
                    yNestedTest = nestedTestDataSet.iloc[:, 0].T

                    yNestedPredicted = nestedClf.predict(xNestedTest)
                    cmat = confusion_matrix(yNestedTest, yNestedPredicted)

                    nestedAccuracy += cmat.diagonal() / cmat.sum(axis=1)

                for c in range(3):
                    if (nestedAccuracy[c] >= highestAccuracy[c]):
                        highestAccuracy[c] = nestedAccuracy[c]
                        bestCs[-1][c] = nestedC

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, 1:]
            yTest = testDataSet.iloc[:, 0].T

            trainAccuracies.append([0.0] * 3)
            trainPrecisions.append([0.0] * 3)
            trainRecalls.append([0.0] * 3)
            accuracies.append([0.0] * 3)
            precisions.append([0.0] * 3)
            recalls.append([0.0] * 3)

            for c in range(3):
                clf = OneVsRestClassifier(
                    SVC(kernel='linear', C=bestCs[-1][c]))
                clf.fit(xTrain, yTrain)

                yTrainPredicted = clf.predict(xTrain)
                trainCmat = confusion_matrix(yTrain, yTrainPredicted)
                trainAccuracy = trainCmat.diagonal() / trainCmat.sum(axis=1)
                trainScore = precision_recall_fscore_support(
                    yTrain, yTrainPredicted)
                trainPrecision = trainScore[0]
                trainRecall = trainScore[1]

                trainAccuracies[-1][c] = trainAccuracy[c]
                trainPrecisions[-1][c] = trainPrecision[c]
                trainRecalls[-1][c] = trainRecall[c]

                yTestPredicted = clf.predict(xTest)
                testCmat = confusion_matrix(yTest, yTestPredicted)
                testAccuracy = testCmat.diagonal() / testCmat.sum(axis=1)
                testScore = precision_recall_fscore_support(
                    yTest, yTestPredicted)
                testPrecision = testScore[0]
                testRecall = testScore[1]

                accuracies[-1][c] = testAccuracy[c]
                precisions[-1][c] = testPrecision[c]
                recalls[-1][c] = testRecall[c]

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                fold, "\t".join(map(str, bestCs[-1])), "\t".join(
                    map(str, trainAccuracies[-1])), "\t".join(
                        map(str, trainPrecisions[-1])), "\t".join(
                            map(str, trainRecalls[-1])), "\t".join(
                                map(str, accuracies[-1])), "\t".join(
                                    map(str, precisions[-1])), "\t".join(
                                        map(str, recalls[-1]))))

            fold += 1

        print("Mean\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            "\t".join(map(str, np.mean(bestCs, axis=0))),
            "\t".join(map(str, np.mean(trainAccuracies, axis=0))), "\t".join(
                map(str, np.mean(trainPrecisions, axis=0))), "\t".join(
                    map(str, np.mean(trainRecalls, axis=0))), "\t".join(
                        map(str, np.mean(accuracies, axis=0))), "\t".join(
                            map(str, np.mean(precisions, axis=0))), "\t".join(
                                map(str, np.mean(recalls, axis=0)))))
        print("Standard Deviation\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            "\t".join(map(str, np.std(bestCs, axis=0))),
            "\t".join(map(str, np.std(trainAccuracies, axis=0))), "\t".join(
                map(str, np.std(trainPrecisions, axis=0))), "\t".join(
                    map(str, np.std(trainRecalls, axis=0))), "\t".join(
                        map(str, np.std(accuracies, axis=0))), "\t".join(
                            map(str, np.std(precisions, axis=0))), "\t".join(
                                map(str, np.std(recalls, axis=0)))))

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

        print("Fold\tC1-C\tC2-C\tC3-C\tC1-Gamma\tC2-Gamma\tC3-Gamma\
            \tC1-TrainAccuracy\tC2-TrainAccuracy\tC3-TrainAccuracy\
            \tC1-TrainPrecision\tC2-TrainPrecision\tC3-TrainPrecision\
            \tC1-TrainRecall\tC2-TrainRecall\tC3-TrainRecall\
            \tC1-TestAccuracy\tC2-TestAccuracy\tC3-TestAccuracy\
            \tC1-TestPrecision\tC2-TestPrecision\tC3-TestPrecision\
            \tC1-TestRecall\tC2-TestRecall\tC3-TestRecall")
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet = self.dataSet.iloc[trainIndex]
            xTrain = trainDataSet.iloc[:, 1:]
            yTrain = trainDataSet.iloc[:, 0].T

            nestedTrainDataSets = []
            nestedTestDataSets = []
            for nestedTrainIndex, nestedTestIndex in self.mf.split(
                    trainDataSet):
                nestedTrainDataSets.append(trainDataSet.iloc[nestedTrainIndex])
                nestedTestDataSets.append(trainDataSet.iloc[nestedTestIndex])

            highestAccuracy = [-1] * 3

            bestCs.append([0.0] * 3)
            bestGammas.append([0.0] * 3)
            for nestedC in self.cList:
                for nestedGamma in self.gammaList:
                    nestedAccuracy = 0.0
                    for i in range(self.mFold):
                        nestedTrainDataSet = nestedTrainDataSets[i]
                        xNestedTrain = nestedTrainDataSet.iloc[:, 1:]
                        yNestedTrain = nestedTrainDataSet.iloc[:, 0].T

                        nestedClf = OneVsRestClassifier(
                            SVC(C=nestedC, gamma=nestedGamma))
                        nestedClf.fit(xNestedTrain, yNestedTrain)

                        nestedTestDataSet = nestedTestDataSets[i]
                        xNestedTest = nestedTestDataSet.iloc[:, 1:]
                        yNestedTest = nestedTestDataSet.iloc[:, 0].T

                        yNestedPredicted = nestedClf.predict(xNestedTest)
                        cmat = confusion_matrix(yNestedTest, yNestedPredicted)

                        nestedAccuracy += cmat.diagonal() / cmat.sum(axis=1)

                    for c in range(3):
                        if (nestedAccuracy[c] >= highestAccuracy[c]):
                            highestAccuracy[c] = nestedAccuracy[c]
                            bestCs[-1][c] = nestedC
                            bestGammas[-1][c] = nestedGamma

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, 1:]
            yTest = testDataSet.iloc[:, 0].T

            trainAccuracies.append([0.0] * 3)
            trainPrecisions.append([0.0] * 3)
            trainRecalls.append([0.0] * 3)
            accuracies.append([0.0] * 3)
            precisions.append([0.0] * 3)
            recalls.append([0.0] * 3)

            for c in range(3):
                clf = OneVsRestClassifier(
                    SVC(C=bestCs[-1][c], gamma=bestGammas[-1][c]))
                clf.fit(xTrain, yTrain)

                yTrainPredicted = clf.predict(xTrain)
                trainCmat = confusion_matrix(yTrain, yTrainPredicted)
                trainAccuracy = trainCmat.diagonal() / trainCmat.sum(axis=1)
                trainScore = precision_recall_fscore_support(
                    yTrain, yTrainPredicted)
                trainPrecision = trainScore[0]
                trainRecall = trainScore[1]

                trainAccuracies[-1][c] = trainAccuracy[c]
                trainPrecisions[-1][c] = trainPrecision[c]
                trainRecalls[-1][c] = trainRecall[c]

                yTestPredicted = clf.predict(xTest)
                testCmat = confusion_matrix(yTest, yTestPredicted)
                testAccuracy = testCmat.diagonal() / testCmat.sum(axis=1)
                testScore = precision_recall_fscore_support(
                    yTest, yTestPredicted)
                testPrecision = testScore[0]
                testRecall = testScore[1]

                accuracies[-1][c] = testAccuracy[c]
                precisions[-1][c] = testPrecision[c]
                recalls[-1][c] = testRecall[c]

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                fold, "\t".join(map(str, bestCs[-1])), "\t".join(
                    map(str, bestGammas[-1])), "\t".join(
                        map(str, trainAccuracies[-1])), "\t".join(
                            map(str, trainPrecisions[-1])), "\t".join(
                                map(str, trainRecalls[-1])), "\t".join(
                                    map(str, accuracies[-1])), "\t".join(
                                        map(str, precisions[-1])), "\t".join(
                                            map(str, recalls[-1]))))

            fold += 1

        print("Mean\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            "\t".join(map(str, np.mean(bestCs, axis=0))), "\t".join(
                map(str, np.mean(bestGammas, axis=0))),
            "\t".join(map(str, np.mean(trainAccuracies, axis=0))), "\t".join(
                map(str, np.mean(trainPrecisions, axis=0))), "\t".join(
                    map(str, np.mean(trainRecalls, axis=0))), "\t".join(
                        map(str, np.mean(accuracies, axis=0))), "\t".join(
                            map(str, np.mean(precisions, axis=0))), "\t".join(
                                map(str, np.mean(recalls, axis=0)))))
        print("Standard Deviation\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            "\t".join(map(str, np.std(bestCs, axis=0))), "\t".join(
                map(str, np.std(bestGammas, axis=0))),
            "\t".join(map(str, np.std(trainAccuracies, axis=0))), "\t".join(
                map(str, np.std(trainPrecisions, axis=0))), "\t".join(
                    map(str, np.std(trainRecalls, axis=0))), "\t".join(
                        map(str, np.std(accuracies, axis=0))), "\t".join(
                            map(str, np.std(precisions, axis=0))), "\t".join(
                                map(str, np.std(recalls, axis=0)))))

    def validate(self):
        if self.linear:
            self.linearSVC()
        else:
            self.rbfSVC()
