from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from .modelResults import ModelResults
from .modelResults import ModelResult


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
