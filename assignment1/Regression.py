from __future__ import division
import math
import Tree
import pandas as pd
from time import time
import numpy


class Regression:
    def __init__(self, dataSet, k, nmins):
        self.dataSet = dataSet
        self.k = k
        self.nmins = nmins

    def normalizeDataSet(self):
        for attribute in self.attributes:
            maxAttributeValue = self.dataSet[attribute].max()
            minAttributeValue = self.dataSet[attribute].min()
            self.dataSet[attribute] = self.dataSet[attribute].apply(
                lambda x: (x - minAttributeValue) / (maxAttributeValue - minAttributeValue)
            )

    def sumOfSquaredErrors(self, dataSet):
        mean = dataSet[self.targetAttribute].mean()
        error = 0.0
        for index, row in dataSet.iterrows():
            error += (row[self.targetAttribute] - mean)**2
        return error

    def splitData(self, dataSet, attribute, value):
        subSets = []
        subSets.append(dataSet.loc[dataSet[attribute] <= value])
        subSets.append(dataSet.loc[dataSet[attribute] > value])
        return subSets

    def getAttributeValues(self, dataSet, attribute):
        attributeValues = set()
        sortedDataSet = dataSet.sort_values(attribute)
        previousValue = sortedDataSet[attribute].iloc[0]
        for index, row in sortedDataSet[1:].iterrows():
            currentValue = row[attribute]
            attributeValues.add((previousValue + currentValue) / 2)
            previousValue = currentValue
        return list(attributeValues)

    def errorReduction(self, dataSet, sse, attribute, value):
        dataSetEntries = dataSet.shape[0]
        error = sse
        for childDataSet in self.splitData(dataSet, attribute, value):
            error -= (childDataSet.shape[0] / dataSetEntries
                      ) * self.sumOfSquaredErrors(childDataSet)
        return error

    def errorReductionWrapper(self, dataSet, sse, attribute):
        attributeValues = self.getAttributeValues(dataSet, attribute)
        attributeValue = attributeValues[0]
        error = self.errorReduction(dataSet, sse, attribute, attributeValue)
        for newAttributeValue in attributeValues[1:]:
            newError = self.errorReduction(dataSet, sse, attribute,
                                           newAttributeValue)
            if newError < error:
                error = newError
                attributeValue = newAttributeValue
        return error, attributeValue

    def getBestAttribute(self, dataSet, sse, attributes):
        bestAttribute = attributes[0]
        error, attributeValue = self.errorReductionWrapper(
            dataSet, sse, bestAttribute)
        for attribute in attributes[1:]:
            newError, newAttributeValue = self.errorReductionWrapper(
                dataSet, sse, attribute)
            if newError < error:
                error = newError
                attributeValue = newAttributeValue
                bestAttribute = attribute
        return bestAttribute, attributeValue

    def isPure(self, dataSet):
        return dataSet[self.targetAttribute].unique().shape[0] == 1

    def buildDecisionTree(self, dataSet, threshold, attributes):
        root = Tree.MultiSplitTree()
        mostCommonLabel = dataSet[self.targetAttribute].mean()
        if self.isPure(dataSet) or len(
                attributes) == 0 or dataSet.shape[0] < threshold:
            root.isLeaf = True
            root.label = mostCommonLabel
            return root
        else:
            root.isTree = True
            root.entropy = self.sumOfSquaredErrors(dataSet)
            root.mostCommonLabel = mostCommonLabel
            attribute, attributeValue = self.getBestAttribute(
                dataSet, root.entropy, attributes)
            root.attribute = attribute
            newAttributes = attributes[:]
            newAttributes.remove(attribute)
            root.value = attributeValue
            subSet = dataSet.loc[dataSet[attribute] <= attributeValue]
            if (subSet.shape[0] > 0):
                treeBranch = self.buildDecisionTree(subSet, threshold,
                                                    newAttributes)
                root.trueBranch = treeBranch
            else:
                treeBranch = Tree.MultiSplitTree()
                treeBranch.isLeaf = True
                treeBranch.label = mostCommonLabel
                root.trueBranch = treeBranch
            subSet = dataSet.loc[dataSet[attribute] > attributeValue]
            if (subSet.shape[0] > 0):
                treeBranch = self.buildDecisionTree(subSet, threshold,
                                                    newAttributes)
                root.falseBranch = treeBranch
            else:
                treeBranch = Tree.MultiSplitTree()
                treeBranch.isLeaf = True
                treeBranch.label = mostCommonLabel
                root.falseBranch = treeBranch
        return root

    def classify(self, decisionTree, data):
        if decisionTree.isLeaf:
            return decisionTree.label
        else:
            value = data[decisionTree.attribute]
            if value <= decisionTree.value:
                return self.classify(decisionTree.trueBranch, data)
            else:
                return self.classify(decisionTree.falseBranch, data)

    def predict(self, decisionTree, dataSet):
        actualValues = []
        predictedValues = []
        for index, row in dataSet.iterrows():
            actualValues.append(row[self.targetAttribute])
            predictedValues.append(self.classify(decisionTree, row))
        error = 0.0
        for i in range(len(actualValues)):
            error += (actualValues[i] - predictedValues[i])**2
        error /= len(actualValues)
        return math.sqrt(error)

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
            setTrainErrors = []
            setTestErrors = []
            start = time()
            for i in range(self.k):
                trainDataSet = pd.concat(
                    dataSets[0:i] + dataSets[i + 1:self.k])
                testDataSet = dataSets[i]
                print("FOLD :: {}".format(i + 1))
                decisionTree = self.buildDecisionTree(trainDataSet, threshold,
                                                      self.attributes)
                setTrainError = self.predict(decisionTree, trainDataSet)
                setTrainErrors.append(setTrainError)
                print("TRAINING SET :: {} :: SSE :: {}".format(
                    trainDataSet.shape[0], setTrainError))
                setTestError = self.predict(decisionTree, testDataSet)
                setTestErrors.append(setTestError)
                print("TEST SET :: {} :: SSE :: {}".format(
                    testDataSet.shape[0], setTestError))
            end = time()
            trainSSE = numpy.mean(setTrainErrors)
            testSSE = numpy.mean(setTestErrors)
            testStandardDeviation = numpy.std(setTestErrors)
            print(
                "TRAIN SSE :: {} :: TEST SSE :: {} :: STANDARD DEVIATION :: {}".
                format(trainSSE, testSSE, testStandardDeviation))
            print("TIME :: {}".format(end - start))
