# k_ = list(range(1, k + 1))
# knn_params = {'classify__n_neighbors': k_}
# knn_params = {'classify__n_neighbors': k_,
#               'classify__leaf_size': [1, 2, 3, 5],
#               'classify__weights': ['uniform', 'distance'],
#               'classify__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


class CustomModels:
    @staticmethod
    def fit_pipelines(data, pipelines, params, model_name, test_data):
        results = pd.DataFrame()
        X = data.drop('class', axis=1)
        Y = data['class']
        X_test = test_data.drop('class', axis=1)
        Y_test = test_data['class']
        scoring = {'accuracy': make_scorer(accuracy_score), 'prec': 'precision',
                   'roc': 'roc_auc', 'recall': 'recall'}

        # Calculating the dummy classifier accuracy
        clf = DummyClassifier(strategy='most_frequent', random_state=0)
        clf.fit(X, Y)
        dummy_score = clf.score(X_test, Y_test)

        prediction_result = pd.DataFrame()

        for pipeline in pipelines.keys():
            print(pipeline)
            model_pipeline = pipelines[pipeline][0]
            grid = GridSearchCV(model_pipeline, params, scoring=scoring, refit='roc')

            # Fit the model
            grid.fit(X, Y)

            # Storing prediction results
            predictions = pd.DataFrame(grid.predict_proba(X_test))
            predictions.columns = ['class_' + str(item) for item in list(grid.classes_)]
            predictions['true_label'] = Y_test
            predictions['predicted_label'] = grid.predict(X_test)
            predictions['model'] = model_name
            predictions['pipeline'] = pipelines[pipeline][1]
            prediction_result = pd.concat([prediction_result, predictions])

            test_score = grid.score(X_test, Y_test)
            result1 = pd.DataFrame(grid.cv_results_)
            result1['best_score'] = grid.best_score_
            result1['pipeline'] = pipelines[pipeline][0]
            result1['pipeline_cleaned'] = pipelines[pipeline][1]
            result1['test_score_on_best_param'] = test_score
            results = pd.concat([results, result1])
        results['dummy_score'] = dummy_score
        results['model'] = model_name
        return results, prediction_result
