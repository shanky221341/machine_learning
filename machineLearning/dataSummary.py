import pandas as pd


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
        res.index = ['col_type', 'count_unique', 'values', 'missing_count']
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
    def returnFrequencyCounts(data, columns):
        f_count = {}
        for column in columns:
            d = data[column]
            df = pd.DataFrame(d.value_counts())
            df['col'] = df.index
            df.columns = ['count', column]
            df = df[[column, 'count']]
            df = df.reset_index()
            df = df.drop(['index'], axis=1)
            f_count[column] = df
        return f_count
