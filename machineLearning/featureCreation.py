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
