from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer  # Row normalization


# missing_strategy = ['mean', 'median', 'complete_case']
# normalization_technique = ['z_score', 'min_max', 'row_norm']


class Pipelines:
    '''
    1. Data -> mean -> KNN
    2. Data -> median -> KNN
    3. Data -> complete_case -> KNN
    4. Data -> replace_with_zero -> KNN
    5. Data -> Z_score_normailze -> mean -> KNN
    6. Data -> Z_score_normailze -> median -> KNN
    7. Data -> Z_score_normailze -> complete_case -> KNN
    8. Data -> complete_case -> Z_score_normailze -> KNN
    9. Data -> min_max -> mean -> KNN
    9. Data -> min_max -> mediann -> KNN
    10. Data -> min_max -> complete_case -> KNN
    '''
    knn = KNeighborsClassifier()
    pipelines = {
        'pipeline1': [Pipeline([('imputer', Imputer(strategy='mean')), ('classify',
                                                                        knn)]), "Data -> impute_mean -> KNN"],
        'pipeline2': [Pipeline([('imputer', Imputer(strategy='median')), ('classify',
                                                                          knn)]), "Data -> impute_median -> KNN"],
        'pipeline_complete_case1': [Pipeline([('classify', knn)]), "Data-> complete_case-> KNN"],
        'pipeline3': [Pipeline([('imputer', Imputer(strategy='mean')), ('standardize',
                                                                        StandardScaler()), ('classify', knn)]),
                      "Data -> impute_mean -> z_score_normalize -> KNN"],
        'pipeline4': [Pipeline([('imputer', Imputer(strategy='median')), ('standardize',
                                                                          StandardScaler()), ('classify', knn)]),
                      "Data -> impute_median -> z_score_normalize -> KNN"],
        'pipeline_complete_case2': [Pipeline([('standardize',
                                               StandardScaler()), ('classify', knn)]),
                                    "Data -> complete_case -> z_score_normalize -> KNN"],
        'pipeline8': [Pipeline([('imputer', Imputer(strategy='mean')), ('standardize',
                                                                        MinMaxScaler()), ('classify', knn)]),
                      "Data -> impute_mean -> min_max_normalize -> KNN"],
        'pipeline9': [Pipeline([('imputer', Imputer(strategy='median')), ('standardize',
                                                                          MinMaxScaler()), ('classify', knn)]),
                      "Data -> impute_median -> min_max_normalize -> KNN"],
        'pipeline_complete_case3': [Pipeline([('standardize',
                                               MinMaxScaler()), ('classify', knn)]),
                                    "Data -> complete_case -> min_max_normalize -> KNN"],
        'pipeline12': [Pipeline([('imputer', Imputer(strategy='mean')), ('normalize',
                                                                         Normalizer()), ('classify', knn)]),
                       "Data -> impute_mean -> row_normalize -> KNN"],
        'pipeline13': [Pipeline([('imputer', Imputer(strategy='median')), ('normalize',
                                                                           Normalizer()), ('classify', knn)]),
                       "Data -> impute_median -> row_normalize -> KNN"],
        'pipeline_complete_case4': [Pipeline([('normalize',
                                               Normalizer()), ('classify', knn)]),
                                    "Data -> complete_case -> row_normalize -> KNN"]
    }
