import pandas as pd
from ID3 import ID3
from Regression import Regression


def importData(fileLocation):
    dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
    dataSet = dataSet.drop_duplicates()
    return dataSet


def transformMushroomTargetAttribute(dataSet, targetAttribute):
    uniqueValues = dataSet[targetAttribute].unique()
    dataSet[targetAttribute].replace(
        uniqueValues, range(len(uniqueValues)), inplace=True)
    return dataSet


if __name__ == '__main__':
    print("Iris Dataset")
    irisFileLocation = 'iris.csv'
    irisDataSet = importData(irisFileLocation)
    irisID3 = ID3(irisDataSet, 10, [0.05, 0.10, 0.15, 0.20], True)
    irisID3.validate()
    print("Spambase Dataset")
    spambaseFileLocation = 'spambase.csv'
    spambaseDataSet = importData(spambaseFileLocation)
    spambaseID3 = ID3(spambaseDataSet, 10, [0.05, 0.10, 0.15, 0.20, 0.25],
                      True)
    spambaseID3.validate()
    print("Mushroom Dataset - Multiway Split")
    mushroomFileLocation = 'mushroom.csv'
    mushroomDataSet = importData(mushroomFileLocation)
    columnsLength = len(mushroomDataSet.columns)
    mushroomDataSet = transformMushroomTargetAttribute(mushroomDataSet,
                                                       columnsLength - 1)
    mushroomMultiwayID3 = ID3(mushroomDataSet, 10, [0.05, 0.10, 0.15], False)
    mushroomMultiwayID3.validate()
    print("Mushroom Dataset - Binary Split")
    mushroomModifiedDataSet = pd.get_dummies(
        data=mushroomDataSet, columns=range(columnsLength - 1))
    targetAttributeColumn = mushroomModifiedDataSet[columnsLength - 1]
    mushroomModifiedDataSet.drop(
        labels=[columnsLength - 1], axis=1, inplace=True)
    mushroomModifiedDataSet.insert(
        len(mushroomModifiedDataSet.columns), columnsLength - 1,
        targetAttributeColumn)
    mushroomBinaryID3 = ID3(mushroomModifiedDataSet, 10, [0.05, 0.10, 0.15],
                            True)
    mushroomBinaryID3.validate()
    print("Housing Dataset")
    housingFileLocation = 'housing.csv'
    housingDataSet = importData(housingFileLocation)
    housingRegression = Regression(housingDataSet, 10,
                                   [0.05, 0.10, 0.15, 0.20])
    housingRegression.validate()