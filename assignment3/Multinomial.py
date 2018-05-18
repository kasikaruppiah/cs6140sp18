from __future__ import division
from collections import Counter
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Multinomial:
    def __init__(self):
        self.priorProbability = {}
        self.conditionalProbability = {}

    def computeWordFrequencies(self, trainData):
        wordFrequencies = {}
        for line in open(trainData).readlines():
            documentId, wordId, count = [int(s) for s in line.split()]
            if wordId not in wordFrequencies:
                wordFrequencies[wordId] = count
            else:
                wordFrequencies[wordId] += count

        return wordFrequencies

    def train(self, trainData, trainLabel, vocabulary):
        self.priorProbability = {}
        self.conditionalProbability = {}

        documentClasses = [int(s) for s in open(trainLabel).read().split()]
        totalDocuments = len(documentClasses)
        classFrequency = dict(Counter(documentClasses))

        self.priorProbability = dict(
            map(lambda x: (x[0], math.log(x[1] / totalDocuments)),
                classFrequency.items()))

        wordFrequencies = {}
        classWordFrequency = {}
        for line in open(trainData).readlines():
            documentId, wordId, count = [int(s) for s in line.split()]
            if wordId in vocabulary:
                documentClass = documentClasses[documentId - 1]

                if documentClass not in classWordFrequency:
                    classWordFrequency[documentClass] = count
                else:
                    classWordFrequency[documentClass] += count

                if documentClass not in wordFrequencies:
                    wordFrequencies[documentClass] = {}
                    wordFrequencies[documentClass][wordId] = {}
                    wordFrequencies[documentClass][wordId][documentId] = count
                else:
                    if wordId not in wordFrequencies[documentClass]:
                        wordFrequencies[documentClass][wordId] = {}
                        wordFrequencies[documentClass][wordId][
                            documentId] = count
                    elif documentId not in wordFrequencies[documentClass][
                            wordId]:
                        wordFrequencies[documentClass][wordId][
                            documentId] = count
                    else:
                        wordFrequencies[documentClass][wordId][
                            documentId] += count

        totalWords = len(vocabulary)
        for documentClass in self.priorProbability.keys():
            self.conditionalProbability[documentClass] = {}
            for wordId in vocabulary:
                self.conditionalProbability[documentClass][wordId] = math.log(
                    (sum(
                        wordFrequencies.get(documentClass, {}).get(
                            wordId, {}).values()) + 1) /
                    (classWordFrequency[documentClass] + totalWords))

    def test(self, testData, vocabulary):
        documentProbabilities = {}
        for line in open(testData).readlines():
            documentId, wordId, count = [int(s) for s in line.split()]
            if documentId not in documentProbabilities:
                documentProbabilities[documentId] = {}
            for documentClass in self.priorProbability.keys():
                if documentClass not in documentProbabilities[documentId]:
                    documentProbabilities[documentId][
                        documentClass] = self.priorProbability[documentClass]
                if wordId in vocabulary:
                    documentProbabilities[documentId][
                        documentClass] += count * self.conditionalProbability[
                            documentClass][wordId]

        return dict(
            map(lambda x: (x[0], max(x[1], key=x[1].get)),
                documentProbabilities.items()))

    def calculateMetrics(self, testLabel, predictions):
        testClasses = [int(s) for s in open(testLabel).read().split()]
        predictedClasses = []
        for documentId in range(1, len(testClasses) + 1):
            predictedClasses.append(predictions[documentId])

        accuracy = accuracy_score(testClasses, predictedClasses)
        precision = precision_score(
            testClasses, predictedClasses, average="weighted")
        recall = recall_score(
            testClasses, predictedClasses, average="weighted")

        return accuracy, precision, recall

    def calculateClassMetrics(self, testLabel, predictions):
        testClasses = [int(s) for s in open(testLabel).read().split()]
        predictedClasses = []
        for documentId in range(1, len(testClasses) + 1):
            predictedClasses.append(predictions[documentId])

        precision = precision_score(
            testClasses, predictedClasses, average=None)
        recall = recall_score(testClasses, predictedClasses, average=None)

        print("Precision::\n{}".format(precision))
        print("Recall::\n{}".format(recall))

    def run(self, trainData, trainLabel, testData, testLabel):
        wordFrequencies = self.computeWordFrequencies(trainData)
        sortedWords = sorted(
            wordFrequencies, key=wordFrequencies.get, reverse=True)

        vocabularySize = [
            100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000,
            len(sortedWords)
        ]
        vocabularySize = [50]

        print("Size\tAccuracy\tPrecision\tRecall")
        for size in vocabularySize:
            vocabulary = sortedWords[:size]
            self.train(trainData, trainLabel, vocabulary)
            predictions = self.test(testData, vocabulary)

            accuracy, precision, recall = self.calculateMetrics(
                testLabel, predictions)
            print("{}\t{}\t{}\t{}".format(size, accuracy, precision, recall))

            if size == len(sortedWords):
                self.calculateClassMetrics(testLabel, predictions)
