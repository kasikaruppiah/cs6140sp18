from __future__ import division
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class ModifiedSVMAUC:
    def __init__(self, dataSet, filename, linear=True, kFold=10, mFold=5):
        rows = int(dataSet.shape[0] / 10)
        # rows = 100
        print(rows)
        self.dataSet = dataSet.head(rows)
        self.filename = filename
        self.linear = linear

        self.kFold = kFold
        self.kf = KFold(n_splits=kFold, shuffle=True)

        self.mFold = mFold
        self.mf = KFold(n_splits=mFold, shuffle=True)

        self.cList = [2**i for i in range(-5, 11)]
        self.gammaList = [2**i for i in range(-15, 6)]

    def linearSVC(self):
        bestCs = []
        accuracies = []

        fold = 1

        print("Fold\tC\tAccuracy")
        plt.figure()
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

                    nestedClf = SVC(
                        kernel='linear', C=nestedC, probability=True)
                    nestedClf.fit(xNestedTrain, yNestedTrain)

                    nestedTestDataSet = nestedTestDataSets[i]
                    xNestedTest = nestedTestDataSet.iloc[:, :-1]
                    yNestedTest = nestedTestDataSet.iloc[:, -1].T

                    nestedProba = nestedClf.predict_proba(xNestedTest)
                    nestedFPR, nestedTPR, nestedThresholds = metrics.roc_curve(
                        yNestedTest, nestedProba[:, 1])
                    nestedAccuracy += metrics.auc(nestedFPR, nestedTPR)

                nestedAccuracy /= self.mFold
                if (nestedAccuracy >= highestAccuracy):
                    highestAccuracy = nestedAccuracy
                    bestC = nestedC

            bestCs.append(bestC)

            clf = SVC(kernel='linear', C=bestC, probability=True)
            clf.fit(xTrain, yTrain)

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, :-1]
            yTest = testDataSet.iloc[:, -1].T

            proba = clf.predict_proba(xTest)
            fpr, tpr, thresholds = metrics.roc_curve(yTest, proba[:, 1])
            accuracy = metrics.auc(fpr, tpr)

            plt.plot(
                fpr, tpr, lw=2, label='Fold %d AUC = %0.2f' % (fold, accuracy))

            accuracies.append(accuracy)

            print("{}\t{}\t{}".format(fold, bestC, accuracy))

            fold += 1

        print("Mean\t{}\t{}".format(np.mean(bestCs), np.mean(accuracies)))
        print("Standard Deviation\t{}\t{}".format(
            np.std(bestCs), np.std(accuracies)))

        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig('SVM_linear_ROC_{}'.format(self.filename))

    def rbfSVC(self):
        bestCs = []
        bestGammas = []
        accuracies = []

        fold = 1

        print("Fold\tC\tGamma\tAccuracy")
        plt.figure()
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

                        nestedClf = SVC(
                            C=nestedC, gamma=nestedGamma, probability=True)
                        nestedClf.fit(xNestedTrain, yNestedTrain)

                        nestedTestDataSet = nestedTestDataSets[i]
                        xNestedTest = nestedTestDataSet.iloc[:, :-1]
                        yNestedTest = nestedTestDataSet.iloc[:, -1].T

                        nestedProba = nestedClf.predict_proba(xNestedTest)
                        nestedFPR, nestedTPR, nestedThresholds = metrics.roc_curve(
                            yNestedTest, nestedProba[:, 1])
                        nestedAccuracy += metrics.auc(nestedFPR, nestedTPR)

                    nestedAccuracy /= self.mFold
                    if (nestedAccuracy >= highestAccuracy):
                        highestAccuracy = nestedAccuracy
                        bestC = nestedC
                        bestGamma = nestedGamma

            bestCs.append(bestC)
            bestGammas.append(bestGamma)

            clf = SVC(C=bestC, gamma=bestGamma, probability=True)
            clf.fit(xTrain, yTrain)

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, :-1]
            yTest = testDataSet.iloc[:, -1].T

            proba = clf.predict_proba(xTest)
            fpr, tpr, thresholds = metrics.roc_curve(yTest, proba[:, 1])
            accuracy = metrics.auc(fpr, tpr)

            plt.plot(
                fpr, tpr, lw=2, label='Fold %d AUC = %0.2f' % (fold, accuracy))

            accuracies.append(accuracy)

            print("{}\t{}\t{}\t{}".format(fold, bestC, bestGamma, accuracy))

            fold += 1

        print("Mean\t{}\t{}\t{}".format(
            np.mean(bestCs), np.mean(bestGammas), np.mean(accuracies)))
        print("Standard Deviation\t{}\t{}\t{}".format(
            np.std(bestCs), np.std(bestGammas), np.std(accuracies)))

        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig('SVM_rbf_ROC_{}'.format(self.filename))

    def validate(self):
        if self.linear:
            self.linearSVC()
        else:
            self.rbfSVC()
