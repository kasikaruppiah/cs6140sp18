import pandas as pd
from GradientDescent import GradientDescent
from NormalEquation import NormalEquation
from PolynomialRegression import PolynomialRegression
from RidgeRegression import RidgeRegression


def importData(fileLocation):
    dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
    dataSet = dataSet.drop_duplicates()
    return dataSet


if __name__ == '__main__':
    housingFileLocation = 'housing.csv'
    housingDataSet = importData(housingFileLocation)
    yacthFileLocation = 'yachtData.csv'
    yacthDataSet = importData(yacthFileLocation)
    concreteFileLocation = 'concreteData.csv'
    concreteDataSet = importData(concreteFileLocation)
    sinTrainFileLocation = 'sinDataTrain.csv'
    sinTrainDataSet = importData(sinTrainFileLocation)
    sinValidationFileLocation = 'sinDataValidation.csv'
    sinValidationDataSet = importData(sinValidationFileLocation)

    print('Housing Dataset - Gradient Descent')
    housingGradientDescent = GradientDescent(housingDataSet, 10, 0.0004, 0.005)
    housingGradientDescent.validate()

    print('Yacht Dataset - Gradient Descent')
    yacthGradientDescent = GradientDescent(yacthDataSet, 10, 0.001, 0.001)
    yacthGradientDescent.validate()

    print('Concrete Dataset - Gradient Descent')
    concreteGradientDescent = GradientDescent(concreteDataSet, 10, 0.0007,
                                              0.0001)
    concreteGradientDescent.validate()

    print('Housing Dataset - Normal Equation')
    housingNormalEquation = NormalEquation(housingDataSet, 10)
    housingNormalEquation.validate()

    print('Yacht Dataset - Normal Equation')
    yacthNormalEquation = NormalEquation(yacthDataSet, 10)
    yacthNormalEquation.validate()

    print('Sinusoid Dataset - Polynomial Regression')
    sinPolynomialRegression = PolynomialRegression(
        sinTrainDataSet, sinValidationDataSet, None,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], False)
    sinPolynomialRegression.validate()

    print('Yacth Dataset - Polynomial Regression')
    yacthPolynomialRegression = PolynomialRegression(yacthDataSet, None, 10,
                                                     [1, 2, 3, 4, 5, 6, 7])
    yacthPolynomialRegression.validate()

    print('Sinusoid Dataset - Ridge Regression')
    sinRidgeRegression = RidgeRegression(
        sinTrainDataSet, 10, [1, 2, 3, 4, 5], [
            0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6,
            2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., 4.2, 4.4, 4.6, 4.8, 5., 5.2, 5.4,
            5.6, 5.8, 6., 6.2, 6.4, 6.6, 6.8, 7., 7.2, 7.4, 7.6, 7.8, 8., 8.2,
            8.4, 8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10.
        ])
    sinRidgeRegression.validate()

    newSinRidgeRegression = RidgeRegression(
        sinTrainDataSet, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9], [
            0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6,
            2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., 4.2, 4.4, 4.6, 4.8, 5., 5.2, 5.4,
            5.6, 5.8, 6., 6.2, 6.4, 6.6, 6.8, 7., 7.2, 7.4, 7.6, 7.8, 8., 8.2,
            8.4, 8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10.
        ])
    newSinRidgeRegression.validate()
