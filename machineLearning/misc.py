import pandas as pd


class Misc:
    @staticmethod
    def renameColumns(data, columns):
        data.rename(columns=columns, inplace=True)

    @staticmethod
    def dropColumns(data, columns):
        return data.drop(columns, axis=1)
