from Perceptron import Perceptron
from PerceptronKernel import PerceptronKernel
from RegularizedLogisticRegression import RegularizedLogisticRegression
from SVM import SVM
from ModifiedSVM import ModifiedSVM
from SVMAUC import SVMAUC
from ModifiedSVMAUC import ModifiedSVMAUC
from SVMMulticlass import SVMMulticlass
from SVMMulticlassAUC import SVMMulticlassAUC
from sklearn.utils import shuffle

import pandas as pd


def importData(fileLocation):
    dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
    dataSet = dataSet.fillna(0)
    dataSet = dataSet.drop_duplicates()
    dataSet = shuffle(dataSet)

    return dataSet


if __name__ == '__main__':
    perceptronDataFileLocation = 'perceptronData.csv'
    perceptronDataDataSet = importData(perceptronDataFileLocation)
    twoSpiralsFileLocation = 'twoSpirals.csv'
    twoSpiralsDataSet = importData(twoSpiralsFileLocation)
    spambaseFileLocation = 'spambase.csv'
    spambaseDataSet = importData(spambaseFileLocation)
    diabetesFileLocation = 'diabetes.csv'
    diabetesDataSet = importData(diabetesFileLocation)
    breastCancerFileLocation = 'breastcancer.csv'
    breastCancerDataSet = importData(breastCancerFileLocation)
    wineFileLocation = 'wine.data'
    wineDataSet = importData(wineFileLocation)
    '''
    print('Perceptron Data Dataset - Perceptron Algorithm')
    perceptronDataPerceptron = Perceptron(perceptronDataDataSet)
    perceptronDataPerceptron.validate()

    for i in (1, 5, 10, 20):
        print(
            'Two Spirals Data Dataset - Dual Perceptron Algorithm - Linear Kernel'
        )
        twoSpiralLinearPerceptron = PerceptronKernel(twoSpiralsDataSet, i)
        twoSpiralLinearPerceptron.validate()
'''
    for i in (1, 5, 10, 20):
        print(
            'Two Spirals Data Dataset - Dual Perceptron Algorithm - Guassian Kernel'
        )
        twoSpiralGuassianPerceptron = PerceptronKernel(twoSpiralsDataSet, i,
                                                       False)
        twoSpiralGuassianPerceptron.validate()
'''
    print('Spambase Dataset - Logistic Regression')
    spambaseLogisticRegression = RegularizedLogisticRegression(
        spambaseDataSet, 'Spambase', 0.75, 0.00001)
    spambaseLogisticRegression.validate()

    print('Pima Indian Diabetes Dataset - Logistic Regression')
    diabetesLogisticRegression = RegularizedLogisticRegression(
        diabetesDataSet, 'Diabetes', 0.1, 0.0000001)
    diabetesLogisticRegression.validate()

    print('Breast Cancer Dataset - Logistic Regression')
    breastCancerLogisticRegression = RegularizedLogisticRegression(
        breastCancerDataSet, 'Cancer', 0.75, 0.00001)
    breastCancerLogisticRegression.validate()

    print('Pima Indian Diabetes Dataset - SVM linear')
    diabetesSVM = SVM(diabetesDataSet)
    diabetesSVM.validate()

    print('Pima Indian Diabetes Dataset - SVM rbf')
    diabetesSVM = SVM(diabetesDataSet, False)
    diabetesSVM.validate()

    print('Breast Cancer Dataset - SVM linear')
    breastCancerSVM = SVM(breastCancerDataSet)
    breastCancerSVM.validate()

    print('Breast Cancer Dataset - SVM rbf')
    breastCancerSVM = SVM(breastCancerDataSet, False)
    breastCancerSVM.validate()

    print('Spambase Dataset - SVM linear')
    spambaseSVM = SVM(spambaseDataSet)
    spambaseSVM.validate()

    print('Spambase Dataset - SVM rbf')
    spambaseSVM = SVM(spambaseDataSet, False)
    spambaseSVM.validate()

    print('Pima Indian Diabetes Dataset - SVM AUC linear')
    diabetesSVM = SVMAUC(diabetesDataSet)
    diabetesSVM.validate()

    print('Pima Indian Diabetes Dataset - SVM AUC rbf')
    diabetesSVMAUC = SVMAUC(diabetesDataSet, False)
    diabetesSVMAUC.validate()

    print('Breast Cancer Dataset - SVM AUC linear')
    breastCancerSVMAUC = SVMAUC(breastCancerDataSet)
    breastCancerSVMAUC.validate()

    print('Breast Cancer Dataset - SVM AUC rbf')
    breastCancerSVMAUC = SVMAUC(breastCancerDataSet, False)
    breastCancerSVMAUC.validate()

    print('Spambase Dataset - SVM AUC linear')
    spambaseSVMAUC = SVMAUC(spambaseDataSet)
    spambaseSVMAUC.validate()

    print('Spambase Dataset - SVM AUC rbf')
    spambaseSVMAUC = SVMAUC(spambaseDataSet, False)
    spambaseSVMAUC.validate()

    print('Wine Dataset - Multiclass SVM linear')
    wineMulticlassSVM = SVMMulticlass(wineDataSet)
    wineMulticlassSVM.validate()

    print('Wine Dataset - Multiclass SVM rbf')
    wineMulticlassSVM = SVMMulticlass(wineDataSet, False)
    wineMulticlassSVM.validate()

    print('Wine Dataset - Multiclass SVM AUC linear')
    wineMulticlassSVMAUC = SVMMulticlassAUC(wineDataSet)
    wineMulticlassSVMAUC.validate()

    print('Wine Dataset - Multiclass SVM AUC rbf')
    wineMulticlassSVMAUC = SVMMulticlassAUC(wineDataSet, False)
    wineMulticlassSVMAUC.validate()
'''
# print('Pima Indian Diabetes Dataset - SVM AUC linear')
# diabetesSVM = ModifiedSVMAUC(diabetesDataSet, 'Diabetes')
# diabetesSVM.validate()

# print('Spambase Dataset - SVM AUC linear')
# spambaseSVMAUC = ModifiedSVMAUC(spambaseDataSet, 'Spambase')
# spambaseSVMAUC.validate()
