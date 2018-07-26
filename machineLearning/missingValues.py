from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin
import pandas as pd


class MissingValue:
    @staticmethod
    def checkVal(val_to_replace, replace_with_val, x):
        if pd.isnull(x):
            if pd.isnull(val_to_replace):
                return replace_with_val
            else:
                return x
        else:
            if x == val_to_replace:
                return replace_with_val
            else:
                return x

    @staticmethod
    def replaceValuesInColumns(data, val_to_replace, replace_with_val, columns):
        tmp = data.copy()
        for col in columns:
            tmp[col] = tmp[col].map(lambda x: MissingValue.checkVal(val_to_replace, replace_with_val, x))
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

    @staticmethod
    def imupteMissingWithMedianValues(data):
        imputer = Imputer(strategy="median")
        imputer.fit(data)
        dataTransformed = pd.DataFrame(imputer.transform(data))
        meanValues = pd.DataFrame(imputer.statistics_)
        meanValues.index = data.columns
        meanValues._columns = ['median']
        dataTransformed.columns = data.columns
        return imputer, meanValues, dataTransformed


class CustomQuantitativeImputer(TransformerMixin):
    """Transformer for imputing missing value on specific columns.
    """

    def __init__(self, cols, strategy='mean'):
        self.d = {}
        self.cols = cols
        self.strategy = strategy

    def transform(self, X, **transform_params):
        """Transforms X to impute specific columns.

        Args:
            X (obj): The dataset to transform. Can be dataframe or matrix.
            transform_params (kwargs, optional): Additional params.

        Returns:
            The transformed dataset with imputed columns.
        """
        data = X.copy()
        for col in self.cols:
            data[col] = self.d[col].transform(data[[col]])
        return data

    def fit(self, X, y=None, **fit_params):
        """Fits transfomer over X.

        Needs to apply fit over the Imputer separately for each col so as to retain the
        label classes when transforming.
        """
        for col in self.cols:
            impute = Imputer(strategy=self.strategy)
            self.d[col] = impute.fit(X[[col]])
        return self


class CustomCategoryImputer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df):
        X = df.copy()
        for col in self.cols:
            X[col].fillna(X[col].value_counts().index[0], inplace=True)
        return X

    def fit(self, *_):
        return self


class CustomDummifier(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, X):
        return pd.get_dummies(X, columns=self.cols)

    def fit(self, *_):
        return self


class CustomEncoder(TransformerMixin):
    def __init__(self, col, ordering=None):
        self.ordering = ordering
        self.col = col

    def transform(self, df):
        X = df.copy()
        X[self.col] = X[self.col].map(lambda x: self.ordering.index(x))
        return X

    def fit(self, *_):
        return self


class CustomEncoderManyCols(TransformerMixin):
    def __init__(self, ordering_dict):
        self.ordering_dict = ordering_dict

    def transform(self, df):
        X = df.copy()
        for col in self.ordering_dict.keys():
            X[col] = X[col].map(lambda x: self.ordering_dict[col].index(x))
        return X

    def fit(self, *_):
        return self
