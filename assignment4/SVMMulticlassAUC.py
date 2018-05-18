from __future__ import division
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import label_binarize

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SVMMulticlassAUC:
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
        FPRs = []
        TPRs = []
        accuracies = []

        fold = 1

        print(
            "Fold\tC1-C\tC2-C\tC3-C\tC1-TestAccuracy\tC2-TestAccuracy\tC3-TestAccuracy"
        )
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet = self.dataSet.iloc[trainIndex]
            xTrain = trainDataSet.iloc[:, 1:]
            yTrain = trainDataSet.iloc[:, 0].T
            yTrain = label_binarize(yTrain, classes=[1, 2, 3])

            nestedTrainDataSets = []
            nestedTestDataSets = []
            for nestedTrainIndex, nestedTestIndex in self.mf.split(
                    trainDataSet):
                nestedTrainDataSets.append(trainDataSet.iloc[nestedTrainIndex])
                nestedTestDataSets.append(trainDataSet.iloc[nestedTestIndex])

            highestAccuracy = [-1] * 3

            bestCs.append([0.0] * 3)
            for nestedC in self.cList:
                nestedAccuracy = [0.0] * 3
                for i in range(self.mFold):
                    nestedTrainDataSet = nestedTrainDataSets[i]
                    xNestedTrain = nestedTrainDataSet.iloc[:, 1:]
                    yNestedTrain = nestedTrainDataSet.iloc[:, 0].T
                    yNestedTrain = label_binarize(
                        yNestedTrain, classes=[1, 2, 3])

                    nestedClf = OneVsRestClassifier(
                        SVC(kernel='linear', C=nestedC, probability=True))
                    nestedClf.fit(xNestedTrain, yNestedTrain)

                    nestedTestDataSet = nestedTestDataSets[i]
                    xNestedTest = nestedTestDataSet.iloc[:, 1:]
                    yNestedTest = nestedTestDataSet.iloc[:, 0].T
                    yNestedTest = label_binarize(
                        yNestedTest, classes=[1, 2, 3])

                    nestedProba = nestedClf.predict_proba(xNestedTest)
                    for c in range(3):
                        nestedFPR, nestedTPR, nestedThresholds = metrics.roc_curve(
                            yNestedTest[:, c], nestedProba[:, c])
                        nestedAccuracy[c] += metrics.auc(nestedFPR, nestedTPR)

                for c in range(3):
                    if (nestedAccuracy[c] >= highestAccuracy[c]):
                        highestAccuracy[c] = nestedAccuracy[c]
                        bestCs[-1][c] = nestedC

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, 1:]
            yTest = testDataSet.iloc[:, 0].T
            yTest = label_binarize(yTest, classes=[1, 2, 3])

            accuracies.append([0.0] * 3)
            cFPR = []
            cTPR = []
            for c in range(3):
                clf = OneVsRestClassifier(
                    SVC(kernel='linear', C=bestCs[-1][c], probability=True))
                clf.fit(xTrain, yTrain)

                proba = clf.predict_proba(xTest)
                fpr, tpr, thresholds = metrics.roc_curve(
                    yTest[:, c], proba[:, c])
                cFPR.append(fpr)
                cTPR.append(tpr)
                accuracies[-1][c] = metrics.auc(fpr, tpr)

            print("{}\t{}\t{}".format(fold, "\t".join(map(str, bestCs[-1])),
                                      "\t".join(map(str, accuracies[-1]))))
            FPRs.append(cFPR)
            TPRs.append(cTPR)

            fold += 1

        print("Mean\t{}\t{}".format("\t".join(
            map(str, np.mean(bestCs, axis=0))), "\t".join(
                map(str, np.mean(accuracies, axis=0)))))
        print("Standard Deviation\t{}\t{}".format("\t".join(
            map(str, np.std(bestCs, axis=0))), "\t".join(
                map(str, np.std(accuracies, axis=0)))))

        for c in range(3):
            plt.figure()
            for i in range(self.kFold):
                plt.plot(
                    FPRs[i][c],
                    TPRs[i][c],
                    lw=2,
                    label='Fold %d AUC = %0.2f' % (i + 1, accuracies[i][c]))
                plt.plot([0, 1], [0, 1], 'r--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.title(
                    'Receiver Operating Characteristic - Class {}'.format(
                        c + 1))
                plt.legend(loc='lower right')
                plt.savefig('SVMMulticlass_linear_ROC_{}'.format(c + 1))

    def rbfSVC(self):
        bestCs = []
        FPRs = []
        TPRs = []
        bestGammas = []
        accuracies = []

        fold = 1

        print("Fold\tC1-C\tC2-C\tC3-C\tC1-Gamma\tC2-Gamma\tC3-Gamma\
            \tC1-TestAccuracy\tC2-TestAccuracy\tC3-TestAccuracy")
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet = self.dataSet.iloc[trainIndex]
            xTrain = trainDataSet.iloc[:, 1:]
            yTrain = trainDataSet.iloc[:, 0].T
            yTrain = label_binarize(yTrain, classes=[1, 2, 3])

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
                    nestedAccuracy = [0.0] * 3
                    for i in range(self.mFold):
                        nestedTrainDataSet = nestedTrainDataSets[i]
                        xNestedTrain = nestedTrainDataSet.iloc[:, 1:]
                        yNestedTrain = nestedTrainDataSet.iloc[:, 0].T
                        yNestedTrain = label_binarize(
                            yNestedTrain, classes=[1, 2, 3])

                        nestedClf = OneVsRestClassifier(
                            SVC(C=nestedC, gamma=nestedGamma,
                                probability=True))
                        nestedClf.fit(xNestedTrain, yNestedTrain)

                        nestedTestDataSet = nestedTestDataSets[i]
                        xNestedTest = nestedTestDataSet.iloc[:, 1:]
                        yNestedTest = nestedTestDataSet.iloc[:, 0].T
                        yNestedTest = label_binarize(
                            yNestedTest, classes=[1, 2, 3])

                        nestedProba = nestedClf.predict_proba(xNestedTest)
                        for c in range(3):
                            nestedFPR, nestedTPR, nestedThresholds = metrics.roc_curve(
                                yNestedTest[:, c], nestedProba[:, c])
                            nestedAccuracy[c] += metrics.auc(
                                nestedFPR, nestedTPR)

                    for c in range(3):
                        if (nestedAccuracy[c] >= highestAccuracy[c]):
                            highestAccuracy[c] = nestedAccuracy[c]
                            bestCs[-1][c] = nestedC
                            bestGammas[-1][c] = nestedGamma

            testDataSet = self.dataSet.iloc[testIndex]
            xTest = testDataSet.iloc[:, 1:]
            yTest = testDataSet.iloc[:, 0].T
            yTest = label_binarize(yTest, classes=[1, 2, 3])

            accuracies.append([0.0] * 3)
            cFPR = []
            cTPR = []
            for c in range(3):
                clf = OneVsRestClassifier(
                    SVC(C=bestCs[-1][c],
                        gamma=bestGammas[-1][c],
                        probability=True))
                clf.fit(xTrain, yTrain)

                proba = clf.predict_proba(xTest)
                fpr, tpr, thresholds = metrics.roc_curve(
                    yTest[:, c], proba[:, c])
                cFPR.append(fpr)
                cTPR.append(tpr)
                accuracies[-1][c] = metrics.auc(fpr, tpr)

            print("{}\t{}\t{}\t{}".format(fold, "\t".join(
                map(str, bestCs[-1])), "\t".join(map(str, bestGammas[-1])),
                                          "\t".join(map(str, accuracies[-1]))))

            FPRs.append(cFPR)
            TPRs.append(cTPR)

            fold += 1

        print("Mean\t{}\t{}\t{}".format("\t".join(
            map(str, np.mean(bestCs, axis=0))), "\t".join(
                map(str, np.mean(bestGammas, axis=0))), "\t".join(
                    map(str, np.mean(accuracies, axis=0)))))
        print("Standard Deviation\t{}\t{}\t{}".format("\t".join(
            map(str, np.std(bestCs, axis=0))), "\t".join(
                map(str, np.std(bestGammas, axis=0))), "\t".join(
                    map(str, np.std(accuracies, axis=0)))))

        for c in range(3):
            plt.figure()
            for i in range(self.kFold):
                plt.plot(
                    FPRs[i][c],
                    TPRs[i][c],
                    lw=2,
                    label='Fold %d AUC = %0.2f' % (i + 1, accuracies[i][c]))
                plt.plot([0, 1], [0, 1], 'r--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.title(
                    'Receiver Operating Characteristic - Class {}'.format(
                        c + 1))
                plt.legend(loc='lower right')
                plt.savefig('SVMMulticlass_rbf_ROC_{}'.format(c + 1))

    def validate(self):
        if self.linear:
            self.linearSVC()
        else:
            self.rbfSVC()
