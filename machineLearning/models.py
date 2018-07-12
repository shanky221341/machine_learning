from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from .modelResults import ModelResults
from .modelResults import ModelResult
from .modelInputs import Data
from .visualizations import Visualization
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer


class Model:
    def __init__(self):
        self.classifier_models = {
            'KNN': Model.fitKNNClassifier
        }

    @staticmethod
    def fitKNNClassifier(X, Y, num_of_NNs=5):
        k = list(range(1, num_of_NNs + 1))
        knn_params = {'n_neighbors': k}
        knn = KNeighborsClassifier()
        grid = GridSearchCV(knn, knn_params)
        grid.fit(X, Y)
        best_k = grid.best_params_
        best_score = grid.best_score_
        return best_score, best_k, grid

    def fitModels(self, models):
        for model in models.keys():
            if model not in self.classifier_models.keys():
                raise ValueError('Model should be either of {}'.format(self.classifier_models.keys()))
            model_results = ModelResults()
            l = []
            if model == 'KNN':
                if models[model].num_of_NNs is not None:
                    best_score, best_k, grid = self.classifier_models[model](models[model].X, models[model].Y,
                                                                             models[model].num_of_NNs)
                    model_results.results[model] = ModelResult('KNN', grid)
                    l.append({'knn_accuracy': best_score, 'best_k': best_k})
                else:
                    best_score, best_k, grid = self.classifier_models[model](models[model].X, models[model].Y)
                    model_results.results[model] = ModelResult('KNN', grid)
                    l.append({'knn_accuracy': best_score, 'best_k': best_k['n_neighbors']})
        result_summary = pd.DataFrame(l)
        result_summary.index = ['KNN']
        return result_summary, model_results

    @staticmethod
    def fitKNNManyStrategy(data, strategies, k):
        warnings.warn("""Note: In this method the missing value is being imputed before splitting
        it into train and test or before cross-validation which violates the fundamental
        principles of machine learning. So the results may be overoptimistic.
        Use Carefully""")
        l = []
        for strategy in strategies:
            dataNew = Data.formData(data, missing_strategy=strategy)
            X = dataNew.drop('class', axis=1)
            Y = dataNew['class']
            best_score, best_k, grid = Model.fitKNNClassifier(X, Y, k)
            # print(grid.cv_results_)
            l.append({'knn_accuracy': best_score, 'best_k': best_k, 'strategy': strategy})
            scores = pd.Series(grid.cv_results_['mean_test_score'])
            scores.index = range(1, k + 1)
            print(" KNN with -> {} Performance".format(strategy))
            Visualization.createBarPlotSingleVar(scores, "Showing KNN performance", "Accuracy")
        result_summary = pd.DataFrame(l)
        result_summary.index = ['KNN', 'KNN', 'KNN']
        return result_summary

    @staticmethod
    def fitKNNManyStrategyUsingPipeline(data, strategies, k):

        '''

        Pipeline format: Impute using Missing Strategy -> Fit KNN

        This function takes following parameter and return the result of the pipeline
        :param data: raw data with missing values
        :param k: number of nearest neighbour
        :param strategies: missing strategy can be {'meann','median','complete_case'}
        :return:
        '''
        k_ = list(range(1, k + 1))
        knn_params = {'classify__n_neighbors': k_}
        # must redefine params to fit the pipeline
        knn = KNeighborsClassifier()
        l = []
        for strategy in strategies:
            if strategy == 'mean':
                impute = Pipeline([('imputer', Imputer(strategy='mean')), ('classify',
                                                                           knn)])
                X = data.drop('class', axis=1)
                Y = data['class']
            elif strategy == 'median':
                impute = Pipeline([('imputer', Imputer(strategy='median')), ('classify',
                                                                             knn)])
                X = data.drop('class', axis=1)
                Y = data['class']
            elif strategy == 'complete_case':
                dataNew = Data.formData(data, missing_strategy=strategy)
                X = dataNew.drop('class', axis=1)
                Y = dataNew['class']
                impute = Pipeline([('classify', knn)])
            else:
                raise ValueError("Invalid Strategy {}".format(strategy))
            grid = GridSearchCV(impute, knn_params)
            grid.fit(X, Y)
            best_k = grid.best_params_
            best_score = grid.best_score_
            l.append({'knn_accuracy': best_score, 'best_k': best_k, 'strategy': strategy})
            scores = pd.Series(grid.cv_results_['mean_test_score'])
            scores.index = range(1, k + 1)
            print(" KNN with -> {} Performance".format(strategy))
            Visualization.createBarPlotSingleVar(scores, "Showing KNN performance", "Accuracy")
        result_summary = pd.DataFrame(l)
        result_summary.index = ['KNN', 'KNN', 'KNN']
        return result_summary

    @staticmethod
    def fit_knn_pipelines(data, pipelines, k):
        '''
        This function takes following parameter and return the result of the pipeline
        :param data: raw data with missing values
        :param k: number of nearest neighbour
        :param pipelines: Defined in the module pipelines.py
        :return:
        '''

        k_ = list(range(1, k + 1))
        knn_params = {'classify__n_neighbors': k_}
        # must redefine params to fit the pipeline
        knn = KNeighborsClassifier()
        l = []
        for pipeline in pipelines.keys():
            if 'complete_case' in pipeline:
                dataNew = Data.formData(data, missing_strategy='complete_case')
                X = dataNew.drop('class', axis=1)
                Y = dataNew['class']
            else:
                X = data.drop('class', axis=1)
                Y = data['class']
            model_pipeline = pipelines[pipeline][0]
            grid = GridSearchCV(model_pipeline, knn_params)
            grid.fit(X, Y)
            best_k = grid.best_params_
            best_score = grid.best_score_
            l.append({'knn_accuracy': best_score, 'best_k': best_k, 'pipeline': pipelines[pipeline][0],
                      'pipeline_cleaned': pipelines[pipeline][1]})
            scores = pd.Series(grid.cv_results_['mean_test_score'])
            scores.index = range(1, k + 1)
            print(" KNN with -> {} Performance".format(pipelines[pipeline][1]))
            Visualization.createBarPlotSingleVar(scores, "Showing KNN performance", "Accuracy")
        result_summary = pd.DataFrame(l)
        result_summary.index = ['KNN'] * len(pipelines.keys())
        return result_summary
