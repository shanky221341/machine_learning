import pandas as pd


class Misc:
    @staticmethod
    def rename_columns(data, columns):
        return data.rename(columns=columns, inplace=True)

    @staticmethod
    def drop_columns(data, columns):
        return data.drop(columns, axis=1)

    @staticmethod
    def explainSetDifference(list_a, list_b):
        train_items_not_int_test = set(list_a) - set(list_b)
        test_items_not_int_train = set(list_b) - set(list_a)
        print("Training items not in test are:{}".format(train_items_not_int_test))
        print("Test items not in train are:{}".format(test_items_not_int_train))

    """
    This function will give you classification report in the form
    of the pandas dataframe
    """

    @staticmethod
    def build_precision_table_ver1(classification_report):
        cols = classification_report.split('\n')[0].strip().split(" ")
        cols = [col for col in cols if col != '']
        cols.insert(0, 'class')

        vals1 = classification_report.split('\n')[2].strip().split(" ")
        vals1 = pd.Series([val for val in vals1 if val != ''])

        vals2 = classification_report.split('\n')[3].strip().split(" ")
        vals2 = pd.Series([val for val in vals2 if val != ''])

        tot = classification_report.split('\n')[5].strip().split(" ")
        tot = pd.Series([val for val in tot if val != ''])
        tot = tot[2:].reset_index()
        report = pd.DataFrame(vals1)
        report['1'] = vals2
        report['2'] = tot[0]
        report = report.transpose()
        report.columns = cols
        return report

    """
    This function will give you classification report in the form
    of the pandas dataframe in the long format
    """

    @staticmethod
    def build_precision_table_ver2(classification_report):
        cols = classification_report.split('\n')[0].strip().split(" ")
        cols = [col for col in cols if col != '']
        cols.insert(0, 'class')

        vals1 = classification_report.split('\n')[2].strip().split(" ")
        vals1 = pd.Series([val for val in vals1 if val != ''])

        vals2 = classification_report.split('\n')[3].strip().split(" ")
        vals2 = pd.Series([val for val in vals2 if val != ''])

        tot = classification_report.split('\n')[5].strip().split(" ")
        tot = pd.Series([val for val in tot if val != ''])
        tot = tot[2:].reset_index()

        report = pd.DataFrame()
        report['1'] = pd.concat([vals1[1:], vals2[1:], tot[0][1:]])
        report = report.transpose()
        report.columns = ['class_0_precision', 'class_0_recall', 'class_0_f1_score', 'class_0_support',
                          'class_1_precision', 'class_1_recall', 'class_1_f1_score', 'class_1_support',
                          'total_precision', 'total_recall', 'total_f1_score', 'total_support']
        return report
