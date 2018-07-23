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
