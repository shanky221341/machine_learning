from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer  # Row normalization


class Params:
    knn_params1 = {'classify__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'classify__leaf_size': [1, 2, 3, 5],
                   'classify__weights': ['uniform', 'distance'],
                   'classify__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    knn_params2 = {'classify__n_neighbors': [1]}


class Pipelines:
    knn = KNeighborsClassifier()
    knn_pipelines = {
        'pipeline1': [Pipeline([('classify',
                                 knn)]), "Data -> KNN"],
        'pipeline2': [Pipeline([('standardize',
                                 StandardScaler()), ('classify', knn)]),
                      "Data  z_score_normalize -> KNN"],
        'pipeline3': [Pipeline([('standardize',
                                 MinMaxScaler()), ('classify', knn)]),
                      "Data  min_max_normalize -> KNN"],
        'pipeline4': [Pipeline([('normalize',
                                 Normalizer()), ('classify', knn)]),
                      "Data -> row_normalize -> KNN"]
    }
