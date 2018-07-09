from sklearn.preprocessing import Imputer
import pandas as pd


class MissingValue:
    @staticmethod
    def replaceValuesInColumns(data, val_to_replace, replace_with_val, columns):
        tmp = data.copy()
        for col in columns:
            tmp[col] = tmp[col].map(lambda x: x if x != val_to_replace else replace_with_val)
        return tmp

    @staticmethod
    def dropMissingValuesInSpecificColumns(data, columns=None, complete=False):
        if complete:
            return data.dropna()
        else:
            return data.dropna(subset=columns, axis=0)

    '''
    This function will impute the missing values with mean
    and return the object of class Imputer(sklearn), meanValues DataFrame and 
    the transformed dataframe. 
    '''

    @staticmethod
    def imupteMissingWithMeanValues(data):
        imputer = Imputer(strategy="mean")
        imputer.fit(data)
        dataTransformed = pd.DataFrame(imputer.transform(data))
        meanValues = pd.DataFrame(imputer.statistics_)
        meanValues.index = data.columns
        meanValues._columns = ['mean']
        dataTransformed.columns = data.columns
        return imputer, meanValues, dataTransformed
