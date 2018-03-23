import pandas as pd
from LogisticRegression import LogisticRegression
from MultivariateBernoulli import MultivariateBernoulli
from Multinomial import Multinomial


def importData(fileLocation):
    dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
    dataSet = dataSet.drop_duplicates()
    return dataSet


if __name__ == '__main__':
    spambaseFileLocation = 'spambase.csv'
    spambaseDataSet = importData(spambaseFileLocation)
    breastCancerFileLocation = 'breastcancer.csv'
    breastCancerDataSet = importData(breastCancerFileLocation)
    diabetesFileLocation = 'diabetes.csv'
    diabetesDataSet = importData(diabetesFileLocation)

    print('Spambase Dataset - Logistic Regression')
    spambaseLogisticRegression = LogisticRegression(spambaseDataSet, 0.75,
                                                    0.00001)
    spambaseLogisticRegression.validate()
    print('Breast Cancer Dataset - Logistic Regression')
    breastCancerLogisticRegression = LogisticRegression(
        breastCancerDataSet, 0.75, 0.00001)
    breastCancerLogisticRegression.validate()
    print('Pima Indian Diabetes Dataset - Logistic Regression')
    diabetesLogisticRegression = LogisticRegression(diabetesDataSet, 0.1,
                                                    0.0000001)
    diabetesLogisticRegression.validate()

    # multivariateBernoulli = MultivariateBernoulli()
    # multivariateBernoulli.run('train.data', 'train.label', 'test.data',
    #                           'test.label')
    # multinomial = Multinomial()
    # multinomial.run('train.data', 'train.label', 'test.data', 'test.label')
