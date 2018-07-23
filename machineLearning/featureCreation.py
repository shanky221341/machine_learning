from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MultiLabelBinarizer
from operator import add
import pandas as pd
import numpy as np


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


class CustomCutter(TransformerMixin):
    """
    Create new categorical feature by cutting numeric feature into bins
    In this you will have to manually create bins, so that it covers all
    the categories.
    Eg:
    cc=CustomCutter(bins=[0, 1500, 3000,4500,6000,7500],col='CNT_CHILDREN_freq',
    labels=['0-1500', '1500-3000', '3000-4500','4500-6000','6000-7500'])
    """

    def __init__(self, col, bins, labels=False):
        self.labels = labels
        self.bins = bins
        self.col = col

    def transform(self, df):
        X = df.copy()
        new_col = self.col + '_binned'
        X[new_col] = pd.cut(X[self.col], bins=self.bins,
                            labels=self.labels)
        X[new_col] = X[new_col].astype(object)
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


class IsNegativeFeature(TransformerMixin):
    """Transformer for creating a new feature with True if negative on specific columns.
    """

    def __init__(self, cols):
        self.cols = cols

    def transform(self, df, **transform_params):
        """Transforms df to create new feature with specific columns.

        Args:
            df (obj): The dataset to transform. Can be dataframe or matrix.
            transform_params (kwargs, optional): Additional params.

        Returns:
            The transformed dataset with new features columns.
        """
        X = df.copy()
        for col in self.cols:
            X.loc[:, col + '_is_negative'] = np.where(X[col] < 0, True, False)
        return X

    def fit(self, *_):
        """There is no fit implementation needed."""
        return self


class CreateMeanLookupFeature(TransformerMixin):
    """
    Transformer for creating a new feature with mean of numerical column based on the mean of categorical column.
    """

    def __init__(self, categorical_column, numerical_column, new_col_name, val_for_unseen_category=999):
        self.categorical_column = categorical_column
        self.numerical_column = numerical_column
        self.lookup = None
        self.val_for_unseen_category = val_for_unseen_category
        self.train_data = None
        self.new_col_name = new_col_name

    def transform(self, X):
        """

        :param X: New data to be transformed as per info learned from the training data.
        :param new_col_name: column based on which feature to create
        :return: X
        """

        data = X.copy()

        if pd.isnull(self.lookup):
            raise ValueError("Run fit method before trying to transform")

        values_not_in_train = list(
            set(data[self.categorical_column]) - set(self.train_data[self.categorical_column]))

        # Add default values for un-seen categories
        if len(values_not_in_train) > 0:
            for new_cat in values_not_in_train:
                self.lookup[new_cat] = self.val_for_unseen_category

        data[self.new_col_name] = data[self.categorical_column].map(lambda x: self.lookup[x])
        return data

    def fit(self, X, *_):
        self.train_data = X
        self.lookup = dict(self.train_data.groupby([self.categorical_column])[self.numerical_column].mean())
        return self


class CreateMedianLookupFeature(TransformerMixin):
    """
    Transformer for creating a new feature with mean of numerical column based on the median of categorical column.
    """

    def __init__(self, categorical_column, numerical_column, new_col_name, val_for_unseen_category=999):
        self.categorical_column = categorical_column
        self.numerical_column = numerical_column
        self.lookup = None
        self.val_for_unseen_category = val_for_unseen_category
        self.train_data = None
        self.new_col_name = new_col_name

    def transform(self, X):
        """

        :param X: New data to be transformed as per info learned from the training data.
        :param new_col_name: column based on which feature to create
        :return: X
        """

        data = X.copy()

        if pd.isnull(self.lookup):
            raise ValueError("Run fit method before trying to transform")

        values_not_in_train = list(
            set(data[self.categorical_column]) - set(self.train_data[self.categorical_column]))

        # Add default values for un-seen categories
        if len(values_not_in_train) > 0:
            for new_cat in values_not_in_train:
                self.lookup[new_cat] = self.val_for_unseen_category

        data[self.new_col_name] = data[self.categorical_column].map(lambda x: self.lookup[x])
        return data

    def fit(self, X, *_):
        self.train_data = X
        self.lookup = dict(self.train_data.groupby([self.categorical_column])[self.numerical_column].median())
        return self


class CreateFrequencyLookupFeature(TransformerMixin):
    """
    Transformer for creating a new feature with frequency of categorical column.
    """

    def __init__(self, categorical_column, new_col_name, val_for_unseen_category=999):
        self.categorical_column = categorical_column
        self.lookup = None
        self.val_for_unseen_category = val_for_unseen_category
        self.train_data = None
        self.new_col_name = new_col_name

    def transform(self, X):
        """

        :param X: New data to be transformed as per info learned from the training data.
        :param new_col_name: column based on which feature to create
        :return: X
        """

        data = X.copy()

        if pd.isnull(self.lookup):
            raise ValueError("Run fit method before trying to transform")

        values_not_in_train = list(
            set(data[self.categorical_column]) - set(self.train_data[self.categorical_column]))

        # Add default values for un-seen categories
        if len(values_not_in_train) > 0:
            for new_cat in values_not_in_train:
                self.lookup[new_cat] = self.val_for_unseen_category

        data[self.new_col_name] = data[self.categorical_column].map(lambda x: self.lookup[x])
        return data

    def fit(self, X, *_):
        self.train_data = X
        self.lookup = dict(self.train_data.groupby([self.categorical_column])[self.categorical_column].count())
        return self


class CreateOneHotEncoding(TransformerMixin):
    """
    Transformer for creating one hot encoding.
    """

    def __init__(self, categorical_column):
        self.categorical_column = categorical_column
        self.mlb = None
        self.train_data = None

    def transform(self, X):
        """

        :param X: New data to be transformed as per info learned from the training data.
        :param new_col_name: column based on which feature to create
        :return: X
        """

        data = X.copy()

        if pd.isnull(self.mlb):
            raise ValueError("Run fit method before trying to transform")

        columns = [self.categorical_column.lower() + '_' + item.lower().replace(" ", "_") for item in
                   list(self.mlb.classes_)]
        return data.join(pd.DataFrame(self.mlb.transform(data[self.categorical_column].map(lambda x: [x])),
                                      columns=columns))

    def fit(self, X, *_):
        self.train_data = X

        mlb = MultiLabelBinarizer()
        self.mlb = mlb.fit(self.train_data[self.categorical_column].map(lambda x: [x]))
        return self
