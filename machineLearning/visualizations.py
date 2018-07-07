import matplotlib.pyplot as plt


class Visualization:
    """
    This function takes two numeric variable and draws scatter plot between them.
    It outputs the result in multiple window size depending on the sample size.
    It also takes option of interactive vs non-interactive mode.
    """

    @staticmethod
    def createScatterPlot():
        pass

    @staticmethod
    def createBoxPlot():
        pass

    '''
    This function plots the histogram showing the
    class separation for a given attribute/feature
    '''

    @staticmethod
    def createHistPlot(data, label, col, zero_meaning, one_meaning):
        plt.hist(data[data[label] == 0][col], 10, alpha=0.5, label=zero_meaning)
        plt.hist(data[data[label] == 1][col], 10, alpha=0.5,
                 label=one_meaning)
        plt.legend(loc='upper right')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title('Histogram of {}'.format(col))
        plt.show()

    @staticmethod
    def createWordCloud():
        pass

    """
    This function takes dataframe, and list of feaures and returns all possible scatter plots between them
    This function uses createScatterPlot function internally.
    """

    @staticmethod
    def createAllPossibleScatterPlots():
        pass
