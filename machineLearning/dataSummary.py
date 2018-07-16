import pandas as pd
import numpy as np


class DataSummary:
    @staticmethod
    def returnSummaryDataFrame(data):
        res = pd.DataFrame(data.dtypes).transpose()
        tmp1 = pd.DataFrame(data.apply(lambda x: x.nunique(), axis=0)).transpose()
        tmp2 = pd.DataFrame(data.apply(lambda x: str(x.unique()), axis=0)).transpose()
        tmp3 = pd.DataFrame(data.apply(lambda x: x.isnull().sum(), axis=0)).transpose()
        res = res.append(tmp1)
        res = res.append(tmp2)
        res = res.append(tmp3)
        res.index = ['col_type', 'count_unique', 'unique_values', 'missing_count']
        a = data.describe()
        a.index = ['count', 'mean', 'std', 'min', '25%', '50%(median)', '75%', 'max']
        res = res.append(a)
        # Find mode of categorical column
        t = []
        for col in data.columns:
            if data[col].dtype == object:
                t.append(data[col].mode().iloc[0])
            else:
                t.append(np.nan)
        t = pd.DataFrame(t).transpose()
        t.columns = data.columns
        t.index = ['mode']
        res = pd.concat([res, t], axis=0)
        return res

    @staticmethod
    def showSummaryDataFrame(data):
        print("Number of observations:", data.shape[0])
        print("Number of attributes:", data.shape[1])
        print("")
        columns = data.columns
        for column in columns:
            d = data[column]
            print("Attribute -> {} -> Distinct Count -> {}  Data Type -> {}".format(column, d.nunique(), d.dtype))
            print("Unqiue Values ->", d.unique())
            print("")

    @staticmethod
    def returnFrequencyCounts(data, columns, normalize=False):
        f_count = {}
        for column in columns:
            d = data[column]
            df = pd.DataFrame(d.value_counts(normalize=normalize))
            df['col'] = df.index
            df.columns = ['count', column]
            df = df[[column, 'count']]
            df = df.reset_index()
            df = df.drop(['index'], axis=1)
            f_count[column] = df
        return f_count
