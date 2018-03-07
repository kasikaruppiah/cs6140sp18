from __future__ import division
import math
import Tree
import pandas as pd
from time import time
import numpy
from pandas_ml import ConfusionMatrix
from collections import Counter


class ID3:
    def __init__(self, dataSet, k, nmins, binarySplit=False):
        self.dataSet = dataSet
        self.k = k
        self.nmins = nmins
        self.binarySplit = binarySplit

    def normalizeDataSet(self):
        if self.binarySplit:
            for attribute in self.attributes:
                maxAttributeValue = self.dataSet[attribute].max()
                minAttributeValue = self.dataSet[attribute].min()
                self.dataSet[attribute] = self.dataSet[attribute].apply(
                    lambda x: (x - minAttributeValue) / (maxAttributeValue - minAttributeValue)
                )

    def entropy(self, dataSet):
        splits = dataSet[self.targetAttribute].value_counts().to_dict()
        dataSetEntries = dataSet.shape[0]
        entropy = 0.0
        for label, count in splits.iteritems():
            frequency = count / dataSetEntries
            entropy -= frequency * math.log(frequency, 2)
        return entropy

    def splitData(self, dataSet, attribute, value=None):
        subSets = []
        if self.binarySplit:
            subSets.append(dataSet.loc[dataSet[attribute] < value])
            subSets.append(dataSet.loc[dataSet[attribute] >= value])
        else:
            for value in dataSet[attribute].unique():
                subSets.append(dataSet.loc[dataSet[attribute] == value])
        return subSets

    def getAttributeValues(self, dataSet, attribute):
        attributeValues = set()
        sortedDataSet = dataSet.sort_values(attribute)
        previousValue = sortedDataSet[attribute].iloc[0]
        previousLabel = sortedDataSet[self.targetAttribute].iloc[0]
        for index, row in sortedDataSet[1:].iterrows():
            currentLabel = row[self.targetAttribute]
            currentValue = row[attribute]
            if currentLabel != previousLabel:
                attributeValues.add((previousValue + currentValue) / 2)
                previousLabel = currentLabel
            previousValue = currentValue
        return list(attributeValues)

    def getInformationGain(self, dataSet, entropy, attribute, value=None):
        dataSetEntries = dataSet.shape[0]
        informationGain = entropy
        for childDataSet in self.splitData(dataSet, attribute, value):
            informationGain -= (childDataSet.shape[0] / dataSetEntries
                                ) * self.entropy(childDataSet)
        return informationGain

    def informationGainWrapper(self, dataSet, entropy, attribute):
        if self.binarySplit:
            attributeValues = self.getAttributeValues(dataSet, attribute)
            attributeValue = attributeValues[0]
            informationGain = self.getInformationGain(
                dataSet, entropy, attribute, attributeValue)
            for newAttributeValue in attributeValues[1:]:
                newInformationGain = self.getInformationGain(
                    dataSet, entropy, attribute, newAttributeValue)
                if newInformationGain > informationGain:
                    informationGain = newInformationGain
                    attributeValue = newAttributeValue
            return informationGain, attributeValue
        else:
            return self.getInformationGain(dataSet, entropy, attribute,
                                           None), 0.0

    def getBestAttribute(self, dataSet, entropy, attributes):
        bestAttribute = attributes[0]
        informationGain, attributeValue = self.informationGainWrapper(
            dataSet, entropy, bestAttribute)
        for attribute in attributes[1:]:
            newInformationGain, newAttributeValue = self.informationGainWrapper(
                dataSet, entropy, attribute)
            if newInformationGain > informationGain:
                informationGain = newInformationGain
                attributeValue = newAttributeValue
                bestAttribute = attribute
        return bestAttribute, attributeValue

    def isPure(self, dataSet):
        return dataSet[self.targetAttribute].unique().shape[0] == 1

    def buildDecisionTree(self, dataSet, threshold, attributes):
        root = Tree.MultiSplitTree()
        mostCommonLabel = dataSet[self.targetAttribute].value_counts().idxmax()
        if self.isPure(dataSet) or len(
                attributes) == 0 or dataSet.shape[0] < threshold:
            root.isLeaf = True
            root.label = mostCommonLabel
            return root
        else:
            root.isTree = True
            root.entropy = self.entropy(dataSet)
            root.mostCommonLabel = mostCommonLabel
            attribute, attributeValue = self.getBestAttribute(
                dataSet, root.entropy, attributes)
            root.attribute = attribute
            newAttributes = attributes[:]
            newAttributes.remove(attribute)
            if self.binarySplit:
                root.value = attributeValue
                subSet = dataSet.loc[dataSet[attribute] < attributeValue]
                if (subSet.shape[0] > 0):
                    treeBranch = self.buildDecisionTree(
                        subSet, threshold, newAttributes)
                    root.trueBranch = treeBranch
                else:
                    treeBranch = Tree.MultiSplitTree()
                    treeBranch.isLeaf = True
                    treeBranch.label = mostCommonLabel
                    root.trueBranch = treeBranch
                subSet = dataSet.loc[dataSet[attribute] >= attributeValue]
                if (subSet.shape[0] > 0):
                    treeBranch = self.buildDecisionTree(
                        subSet, threshold, newAttributes)
                    root.falseBranch = treeBranch
                else:
                    treeBranch = Tree.MultiSplitTree()
                    treeBranch.isLeaf = True
                    treeBranch.label = mostCommonLabel
                    root.falseBranch = treeBranch
            else:
                for value in dataSet[attribute].unique():
                    subSet = dataSet.loc[dataSet[attribute] == value]
                    treeBranch = self.buildDecisionTree(
                        subSet, threshold, newAttributes)
                    treeBranch.value = value
                    root.branches.append(treeBranch)
        return root

    def classify(self, decisionTree, data):
        if decisionTree.isLeaf:
            return decisionTree.label
        else:
            value = data[decisionTree.attribute]
            if self.binarySplit:
                if value < decisionTree.value:
                    return self.classify(decisionTree.trueBranch, data)
                else:
                    return self.classify(decisionTree.falseBranch, data)
            else:
                treeBranches = filter(lambda x: x.value == value,
                                      decisionTree.branches)
                if len(treeBranches) == 0:
                    return decisionTree.mostCommonLabel
                else:
                    return self.classify(treeBranches[0], data)

    def predict(self, decisionTree, dataSet, test=False):
        error = 0
        for index, row in dataSet.iterrows():
            actual = row[self.targetAttribute]
            prediction = self.classify(decisionTree, row)
            if test:
                self.actualValues.append(actual)
                self.predictedValues.append(prediction)
            if prediction != actual:
                error += 1
        return error

    def calculateConfusionMatrixStats(self):
        confusion_matrix = ConfusionMatrix(self.actualValues,
                                           self.predictedValues)
        confusion_matrix.print_stats()

    def validate(self):
        columns = list(self.dataSet)
        self.attributes = columns[0:len(columns) - 1]
        self.targetAttribute = columns[len(columns) - 1]
        self.normalizeDataSet()
        rows = self.dataSet.shape[0]
        interval = rows // self.k
        shuffledDataSet = self.dataSet.sample(frac=1)
        dataSets = []
        start = 0
        end = 0
        for i in range(self.k):
            start = i * interval
            if (i + 2) * interval <= rows:
                end = (i + 1) * interval
            else:
                end = rows
            dataSets.append(shuffledDataSet[start:end])
        for nmin in self.nmins:
            threshold = nmin * rows
            print("NMIN :: {}".format(nmin))
            setTrainAccuracies = []
            setTestAccuracies = []
            self.actualValues = []
            self.predictedValues = []
            start = time()
            for i in range(self.k):
                trainDataSet = pd.concat(
                    dataSets[0:i] + dataSets[i + 1:self.k])
                testDataSet = dataSets[i]
                print("FOLD :: {}".format(i + 1))
                decisionTree = self.buildDecisionTree(trainDataSet, threshold,
                                                      self.attributes)
                setTrainError = self.predict(decisionTree, trainDataSet, False)
                setTrainAccuracy = (trainDataSet.shape[0] - setTrainError
                                    ) * 100 / trainDataSet.shape[0]
                setTrainAccuracies.append(setTrainAccuracy)
                print("TRAINING SET :: {} :: ERROR :: {} :: ACCURACY :: {}".
                      format(trainDataSet.shape[0], setTrainError,
                             setTrainAccuracy))
                setTestError = self.predict(decisionTree, testDataSet, True)
                setTestAccuracy = (testDataSet.shape[0] - setTestError
                                   ) * 100 / testDataSet.shape[0]
                setTestAccuracies.append(setTestAccuracy)
                print("TEST SET :: {} :: ERROR :: {} :: ACCURACY :: {}".format(
                    testDataSet.shape[0], setTestError, setTestAccuracy))
            end = time()
            trainAccuracy = numpy.mean(setTrainAccuracies)
            testAccuracy = numpy.mean(setTestAccuracies)
            testStandardDeviation = numpy.std(setTestAccuracies)
            print(
                "TRAIN ACCURACY :: {} :: TEST ACCURACY :: {} :: STANDARD DEVIATION :: {}".
                format(trainAccuracy, testAccuracy, testStandardDeviation))
            self.calculateConfusionMatrixStats()
            print("TIME :: {}".format(end - start))
