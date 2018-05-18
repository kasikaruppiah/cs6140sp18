from KMeansClustering import KMeansClustering
from GMMClustering import GMMClustering
from sklearn.utils import shuffle
from scipy.stats import zscore

import pandas as pd


def importData(fileLocation):
    dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
    dataSet = dataSet.fillna(0)
    dataSet.iloc[:, :-1] = dataSet.iloc[:, :-1].apply(zscore)
    dataSet = dataSet.fillna(0)
    dataSet = dataSet.drop_duplicates()
    dataSet = shuffle(dataSet)

    return dataSet


if __name__ == '__main__':
    dermatologyFileLocation = 'dermatologyData.csv'
    dermatologyDataSet = importData(dermatologyFileLocation)
    vowelsFileLocation = 'vowelsData.csv'
    vowelsDataSet = importData(vowelsFileLocation)
    glassFileLocation = 'glassData.csv'
    glassDataSet = importData(glassFileLocation)
    ecoliFileLocation = 'ecoliData.csv'
    ecoliDataSet = importData(ecoliFileLocation)
    yeastFileLocation = 'yeastData.csv'
    yeastDataSet = importData(yeastFileLocation)
    soybeanFileLocation = 'soybeanData.csv'
    soybeanDataSet = importData(soybeanFileLocation)

    # print('Dermatology Data Dataset - K-Means Clurstering')
    # dermatologyKMeansClustering = KMeansClustering(dermatologyDataSet,
    #                                                'Dermatology', range(1, 13))
    # dermatologyKMeansClustering.validate()

    # print('Vowels Data Dataset - K-Means Clurstering')
    # vowelsKMeansClustering = KMeansClustering(vowelsDataSet, 'Vowels',
    #                                           range(1, 23))
    # vowelsKMeansClustering.validate()

    # print('Glass Data Dataset - K-Means Clurstering')
    # glassKMeansClustering = KMeansClustering(glassDataSet, 'Glass', range(
    #     1, 13))
    # glassKMeansClustering.validate()

    # print('Ecoli Data Dataset - K-Means Clurstering')
    # ecoliKMeansClustering = KMeansClustering(ecoliDataSet, 'Ecoli', range(
    #     1, 11))
    # ecoliKMeansClustering.validate()

    # print('Yeast Data Dataset - K-Means Clurstering')
    # yeastKMeansClustering = KMeansClustering(yeastDataSet, 'Yeast', range(
    #     1, 19))
    # yeastKMeansClustering.validate()

    # print('Soybean Data Dataset - K-Means Clurstering')
    # soybeanKMeansClustering = KMeansClustering(soybeanDataSet, 'Soybean',
    #                                            range(1, 31))
    # soybeanKMeansClustering.validate()

    print('Dermatology Data Dataset - GMM Clurstering')
    dermatologyGMMClustering = GMMClustering(dermatologyDataSet, 'Dermatology',
                                             range(1, 13))
    dermatologyGMMClustering.validate()

    # print('Vowels Data Dataset - GMM Clurstering')
    # vowelsGMMClustering = GMMClustering(vowelsDataSet, 'Vowels', range(1, 23))
    # vowelsGMMClustering.validate()

    # print('Glass Data Dataset - GMM Clurstering')
    # glassGMMClustering = GMMClustering(glassDataSet, 'Glass', range(1, 13))
    # glassGMMClustering.validate()

    # print('Ecoli Data Dataset - GMM Clurstering')
    # ecoliGMMClustering = GMMClustering(ecoliDataSet, 'Ecoli', range(1, 11))
    # ecoliGMMClustering.validate()

    # print('Yeast Data Dataset - GMM Clurstering')
    # yeastGMMClustering = GMMClustering(yeastDataSet, 'Yeast', range(1, 19))
    # yeastGMMClustering.validate()

    # print('Soybean Data Dataset - GMM Clurstering')
    # soybeanGMMClustering = GMMClustering(soybeanDataSet, 'Soybean', range(
    #     1, 31))
    # soybeanGMMClustering.validate()
