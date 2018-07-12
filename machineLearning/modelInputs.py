from .missingValues import MissingValue


class Data:
    @staticmethod
    def formData(data, missing_strategy):
        if missing_strategy == 'mean':
            imputer, meanValues, dataTransformed = MissingValue.imupteMissingWithMeanValues(data=data)
        elif missing_strategy == 'median':
            imputer, meanValues, dataTransformed = MissingValue.imupteMissingWithMedianValues(data=data)
        elif missing_strategy == 'complete_case':
            dataTransformed = MissingValue.dropMissingValuesInSpecificColumns(data=data, complete=True)
        return dataTransformed


class KNNInputs:
    def __init__(self, X, Y, num_of_NNs=None):
        self.X = X
        self.Y = Y
        self.num_of_NNs = num_of_NNs
