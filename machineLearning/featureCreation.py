from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from operator import add
import pandas as pd


class FeatureCreation:
    '''
    Outputs result based on the map.
    '''

    @staticmethod
    def mapValue(value, condition):
        for key in condition.keys():
            k = key.replace('value', str(value))
            if eval(k):
                return condition[key]

    '''
    This function will take a dataframe(which contains the categorical feature and its count),mapping condition as inputs.
    Outputs new feature based on the map.
    Example condition ->
    d={'value>5':'high_pop','2 <= value <= 5':'mid_pop','value<2':'low_pop'}
    '''

    @staticmethod
    def createFeatureFromFrequencyCount(dataFrame, condition):
        new_feature = dataFrame['count'].map(lambda value: FeatureCreation.mapValue(value, condition))
        return new_feature

    '''
    This function takes one categorical feature and one numeric
    feature and calculates mean of numeric for each category.        
    '''

    @staticmethod
    def createFeatureFromNumVarBasedOnCatVar():
        pass

    '''
    Create new categorical feature by cutting numeric feature into bins    
    '''


class CustomCutter(TransformerMixin):
    def __init__(self, col, bins, labels=False):
        self.labels = labels
        self.bins = bins
        self.col = col

    def transform(self, df):
        X = df.copy()
        new_col = self.col + '_binned'
        X[new_col] = pd.cut(X[self.col], bins=self.bins,
                            labels=self.labels)
        return X

    def fit(self, *_):
        return self


class CustomPolyomialFeatures(TransformerMixin):
    def __init__(self, cols, degree, include_bias, interaction_only):
        self.cols = cols
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def transform(self, df):
        poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias,
                                  interaction_only=self.interaction_only)

        x_poly = poly.fit_transform(df[self.cols])
        a = poly.get_feature_names()
        l1 = ['x'] * len(df.columns)
        l2 = [str(x) for x in list(range(0, len(df.columns)))]
        pol_feats = list(map(add, l1, l2))
        dictionary = dict(zip(pol_feats, df.columns))
        new_names = []
        for item in a:
            for key in dictionary.keys():
                if key in item:
                    item = item.replace(key, dictionary[key])
            new_names.append(item)
        x_poly_df = pd.DataFrame(x_poly, columns=new_names)
        rem_cols = list(set(df.columns) - set(self.cols))
        df_transformed = pd.concat([x_poly_df, df[rem_cols]], axis=1)
        return df_transformed

    def fit(self, *_):
        return self
