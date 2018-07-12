import matplotlib.pyplot as plt
import pandas as pd


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
    def createHistPlotForVarForBinaryClass(data, label_column, col, zero_meaning, one_meaning):
        plt.hist(data[data[label_column] == 0][col], 10, alpha=0.5, label=zero_meaning)
        plt.hist(data[data[label_column] == 1][col], 10, alpha=0.5,
                 label=one_meaning)
        plt.legend(loc='upper right')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title('Histogram of {}'.format(col))
        plt.show()

    @staticmethod
    def createHistPlotForVarsForBinaryClass(data, label_column, columns, zero_meaning, one_meaning):
        for col in columns:
            Visualization.createHistPlotForVarForBinaryClass(data, label_column, col, zero_meaning, one_meaning)

    '''
    This method creates a bar plot for single variable
    Example:
    See the change in the avg values of the columns after
    removing missing columns
    ax=(droppedData.mean() - dataNew.mean()) /dataNew.mean()
    '''

    @staticmethod
    def createBarPlotSingleVar(pandasSeries, title, y_label):
        ax = pandasSeries.plot(kind='bar',
                               title=title)
        ax.set_ylabel(y_label)
        plt.show()

    @staticmethod
    def createWordCloud():
        pass

    @staticmethod
    def createSingleVarHistPlots(data, columns, size_h=15, size_w=15, sharex=False, sharey=False):
        data[columns].hist(figsize=(size_h, size_w), sharex=sharex, sharey=sharey)
        plt.show()

    @staticmethod
    def visualizeModels(modelResults):
        results = modelResults.results
        for model in results.keys():
            if model == 'KNN':
                grid = results[model].modelResult
                scores = pd.Series(grid.cv_results_['mean_test_score'])
                k = len(grid.cv_results_['params'])
                scores.index = range(1, k + 1)
                print("{} Performance".format(model))
                Visualization.createBarPlotSingleVar(scores, "Showing KNN performance", "Accuracy")
            else:
                print("Model not found to visualize")

    """
    This function takes dataframe, and list of feaures and returns all possible scatter plots between them
    This function uses createScatterPlot function internally.
    """

    @staticmethod
    def createAllPossibleScatterPlots():
        pass
