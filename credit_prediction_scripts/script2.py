import pandas as pd

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.set_option('display.max_colwidth', -1)

from machineLearning.missingValues import MissingValue
from machineLearning.featureCreation import CreateFrequencyLookupFeature
from machineLearning.featureCreation import CreateOneHotEncoding
from machineLearning.featureCreation import IsMissingFeature
from sklearn.metrics import roc_auc_score
from machineLearning.missingValues import CustomQuantitativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

application_train = pd.read_csv("/home/cdsw/data/train_sample1.csv")

application_newData = pd.read_csv("/home/cdsw/data/application_test.csv")


X = application_train.values
Y = application_train['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

X_train = pd.DataFrame(X_train)
X_train.columns = application_train.columns
X_test = pd.DataFrame(X_test)
X_test.columns = application_train.columns

raw_columns = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
               'AMT_ANNUITY', 'AMT_GOODS_PRICE',
               'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
               'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
               'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'HOUR_APPR_PROCESS_START',
               'DAYS_LAST_PHONE_CHANGE',
               'LIVINGAPARTMENTS_AVG',
               'AMT_REQ_CREDIT_BUREAU_DAY',
               'FLOORSMAX_MODE',
               'COMMONAREA_MEDI',
               'NONLIVINGAREA_AVG',
               'NONLIVINGAPARTMENTS_MODE',
               'ELEVATORS_MEDI',
               'COMMONAREA_MEDI',
               'FLOORSMAX_MODE',
               'YEARS_BUILD_MEDI',
               'APARTMENTS_MEDI',
               'NONLIVINGAREA_MEDI',
               'LANDAREA_MEDI',
               'NONLIVINGAPARTMENTS_MODE',
               'LIVINGAPARTMENTS_MODE',
               'YEARS_BUILD_MODE',
               'LANDAREA_MODE',
               'FLOORSMIN_AVG',
               'DEF_60_CNT_SOCIAL_CIRCLE',
               'OBS_60_CNT_SOCIAL_CIRCLE',
               'YEARS_BEGINEXPLUATATION_MODE',
               'NONLIVINGAPARTMENTS_MEDI',
               'NONLIVINGAREA_AVG',
               'YEARS_BUILD_MODE',
               'YEARS_BUILD_AVG',
               'AMT_REQ_CREDIT_BUREAU_YEAR',
               'APARTMENTS_MODE',
               'COMMONAREA_MODE',
               'LIVINGAREA_AVG',
               'EXT_SOURCE_2',
               'LANDAREA_AVG',
               'ENTRANCES_MODE',
               'FLOORSMAX_MEDI',
               'NONLIVINGAPARTMENTS_AVG',
               'APARTMENTS_AVG',
               'YEARS_BEGINEXPLUATATION_MEDI',
               'TOTALAREA_MODE',
               'EXT_SOURCE_3',
               'OBS_30_CNT_SOCIAL_CIRCLE',
               'BASEMENTAREA_AVG',
               'FLOORSMAX_MODE',
               'DEF_30_CNT_SOCIAL_CIRCLE',
               'FLOORSMIN_MODE',
               'LIVINGAPARTMENTS_MEDI',
               'BASEMENTAREA_MODE',
               'AMT_REQ_CREDIT_BUREAU_HOUR',
               'ELEVATORS_AVG',
               'ENTRANCES_MEDI',
               'LIVINGAREA_MODE',
               'EXT_SOURCE_1',
               'ENTRANCES_AVG',
               'FLOORSMIN_MEDI',
               'YEARS_BEGINEXPLUATATION_AVG',
               'FLOORSMAX_AVG',
               'CNT_FAM_MEMBERS',
               'BASEMENTAREA_MEDI',
               'LIVINGAREA_MEDI',
               'AMT_REQ_CREDIT_BUREAU_WEEK',
               'AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT',
               'ELEVATORS_MODE',
               'COMMONAREA_AVG',
               'NONLIVINGAREA_MODE',
               'BASEMENTAREA_AVG'
               ]
raw_columns = list(set(raw_columns))
miss_columns = raw_columns

freq_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_TYPE_SUITE',
                'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                'FLAG_MOBIL',
                'FLAG_EMP_PHONE',
                'FLAG_WORK_PHONE',
                'FLAG_CONT_MOBILE',
                'FLAG_PHONE',
                'FLAG_EMAIL',
                'FLAG_DOCUMENT_2',
                'FLAG_DOCUMENT_3',
                'FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_5',
                'FLAG_DOCUMENT_6',
                'FLAG_DOCUMENT_7',
                'FLAG_DOCUMENT_8',
                'FLAG_DOCUMENT_9',
                'FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11',
                'FLAG_DOCUMENT_12',
                'FLAG_DOCUMENT_13',
                'FLAG_DOCUMENT_14',
                'FLAG_DOCUMENT_15',
                'FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17',
                'FLAG_DOCUMENT_18',
                'FLAG_DOCUMENT_19',
                'FLAG_DOCUMENT_20',
                'FLAG_DOCUMENT_21',
                'WALLSMATERIAL_MODE',
                'FONDKAPREMONT_MODE',
                'OCCUPATION_TYPE',
                'HOUSETYPE_MODE',
                'EMERGENCYSTATE_MODE',
                'WEEKDAY_APPR_PROCESS_START',
                'ORGANIZATION_TYPE',
                'REGION_RATING_CLIENT_W_CITY',
                'REG_REGION_NOT_WORK_REGION',
                'REGION_RATING_CLIENT',
                'REG_CITY_NOT_WORK_CITY',
                'LIVE_CITY_NOT_WORK_CITY',
                'REG_REGION_NOT_LIVE_REGION',
                'LIVE_REGION_NOT_WORK_REGION'
                ]

tmp_removal = ['EMERGENCYSTATE_MODE',
               'FONDKAPREMONT_MODE',
               'HOUSETYPE_MODE',
               'NAME_TYPE_SUITE',
               'OCCUPATION_TYPE',
               'WALLSMATERIAL_MODE']

freq_columns = list(set(freq_columns) - set(tmp_removal))

one_hot_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                   'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
                   'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                   'FLAG_MOBIL',
                   'NAME_INCOME_TYPE',
                   'FLAG_EMP_PHONE',
                   'FLAG_WORK_PHONE',
                   'FLAG_CONT_MOBILE',
                   'FLAG_PHONE',
                   'FLAG_EMAIL',
                   'FLAG_DOCUMENT_2',
                   'FLAG_DOCUMENT_3',
                   'FLAG_DOCUMENT_4',
                   'FLAG_DOCUMENT_5',
                   'FLAG_DOCUMENT_6',
                   'FLAG_DOCUMENT_7',
                   'FLAG_DOCUMENT_8',
                   'FLAG_DOCUMENT_9',
                   'FLAG_DOCUMENT_10',
                   'FLAG_DOCUMENT_11',
                   'FLAG_DOCUMENT_12',
                   'FLAG_DOCUMENT_13',
                   'FLAG_DOCUMENT_14',
                   'FLAG_DOCUMENT_15',
                   'FLAG_DOCUMENT_16',
                   'FLAG_DOCUMENT_17',
                   'FLAG_DOCUMENT_18',
                   'FLAG_DOCUMENT_19',
                   'FLAG_DOCUMENT_20',
                   'FLAG_DOCUMENT_21',
                   'WALLSMATERIAL_MODE',
                   'FONDKAPREMONT_MODE',
                   'OCCUPATION_TYPE',
                   'HOUSETYPE_MODE',
                   'EMERGENCYSTATE_MODE',
                   'WEEKDAY_APPR_PROCESS_START',
                   'ORGANIZATION_TYPE',
                   'REGION_RATING_CLIENT_W_CITY',
                   'REG_REGION_NOT_WORK_REGION',
                   'REGION_RATING_CLIENT',
                   'REG_CITY_NOT_WORK_CITY',
                   'LIVE_CITY_NOT_WORK_CITY',
                   'REG_REGION_NOT_LIVE_REGION',
                   'LIVE_REGION_NOT_WORK_REGION',
                   'REG_CITY_NOT_LIVE_CITY'
                   ]

one_hot_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                   'FLAG_OWN_REALTY', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
cat_miss_columns = ['OCCUPATION_TYPE']


def build_model(X_train, standardize_input=1):
    for col in raw_columns:
        X_train[col] = X_train[col].astype('float64')

    for col in raw_columns:
        X_test[col] = X_test[col].astype('float64')

    global has_fitted_the_main_pipelines
    has_fitted_the_main_pipelines = 0

    def create_all_features():
        # all_pipes will contain pipeline of different feature categories
        all_pipes = []

        # is missing feature
        is_missing_pipes = []
        is_missing_pipes.append(('is_miss_features', IsMissingFeature(cols=miss_columns)))

        # Imputing Missing values for numerical columns using "mean"
        quant_miss_pipes = []
        quant_miss_pipes.append(('miss_impute_cols', CustomQuantitativeImputer(cols=raw_columns, strategy="median")))

        # Creating freqeuncy feature pipelines
        freq_pipes = []
        for col in freq_columns:
            freq_pipes.append(('freq_by_' + col, CreateFrequencyLookupFeature(categorical_column=col, new_col_name=
            col + '_freq')))
        # Creating one-hot feature pipelines
        one_hot_pipes = []
        for col in one_hot_columns:
            one_hot_pipes.append(('one_hot_' + col, CreateOneHotEncoding(categorical_column=col)))

        # Create high level pipeline for all feature categories
        all_pipes.append(Pipeline(is_missing_pipes))
        all_pipes.append(Pipeline(quant_miss_pipes))
        all_pipes.append(Pipeline(freq_pipes))
        all_pipes.append(Pipeline(one_hot_pipes))

        all_pipelines = []
        count = 1
        for pipe in all_pipes:
            all_pipelines.append(('feature_set' + str(count), pipe))
            count += 1

        return Pipeline(all_pipelines)

    final_pipeline = create_all_features()

    # Data Preparer function -> should prepare data for all train, test and new data. Because
    # every data kind(train, test and new data) goes through the same preparation
    # phase

    def dataPreparer(data, has_fitted_the_main_pipelines, final_pipeline, data_type):
        newData = data.copy()
        colsA = newData.columns

        # It is observed that there are few vals with 'XNA' code and replacing them with 'F'
        newData = MissingValue.replaceValuesInColumns(columns=['OCCUPATION_TYPE'], data=newData,
                                                      replace_with_val='Laborers', val_to_replace=None)

        if has_fitted_the_main_pipelines == 0:
            final_pipeline = final_pipeline.fit(newData)
            has_fitted_the_main_pipelines = 1
        newData = final_pipeline.transform(newData)
        colsB = newData.columns
        columns = list(colsB[len(colsA):len(colsB)])
        columns.extend(raw_columns)
        if data_type != 'new_data':
            columns.append('TARGET')
        return newData[columns], has_fitted_the_main_pipelines

    X_train['TARGET'] = X_train['TARGET'].astype('int')
    X_test['TARGET'] = X_test['TARGET'].astype('int')
    '''
    This should be underlying order
    '''
    X_train_prepared, has_fitted_the_main_pipelines = dataPreparer(X_train, has_fitted_the_main_pipelines,
                                                                   final_pipeline, 'train')
    X_test_prepared, has_fitted_the_main_pipelines = dataPreparer(X_test, has_fitted_the_main_pipelines, final_pipeline,
                                                                  'test')
    new_data_prepared, has_fitted_the_main_pipelines = dataPreparer(application_newData, has_fitted_the_main_pipelines,
                                                                    final_pipeline, 'new_data')

    col_list = X_train_prepared.columns
    if standardize_input == 1:
        standardiser = StandardScaler()
        X_train_target = X_train_prepared['TARGET']
        X_test_target = X_test_prepared['TARGET']

        standardiser.fit(X_train_prepared.iloc[:, :-1])
        X_train_prepared = pd.DataFrame(standardiser.transform(X_train_prepared.iloc[:, :-1]))
        X_train_prepared['TARGET'] = X_train_target
        X_test_prepared = pd.DataFrame(standardiser.transform(X_test_prepared.iloc[:, :-1]))
        X_test_prepared['TARGET'] = X_test_target
        new_data_prepared = pd.DataFrame(standardiser.transform(new_data_prepared))

    X_train_prepared.columns = col_list
    X_test_prepared.columns = col_list
    new_data_prepared.columns = col_list[:-1]

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    def showClassificationReport(prediction_result):
        print(
            "This function will generate the classification report for each pipelines best parameter fit classification report:\n")
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n")
        pipelines = set(prediction_result['pipeline'])

        for pipeline in pipelines:
            print("Model ->", set(prediction_result[prediction_result['pipeline'] == pipeline]['model']))
            print("Pipeline ->", pipeline)
            print("\nBest Param ->", set(prediction_result[prediction_result['pipeline'] == pipeline]['best_param']))
            print("===============================================================================================")
            y_pred = prediction_result[prediction_result['pipeline'] == pipeline]['predicted_label']
            y_true = prediction_result[prediction_result['pipeline'] == pipeline]['true_label']
            y_scores = prediction_result[prediction_result['pipeline'] == pipeline]['class_1']
            auc = roc_auc_score(y_true, y_scores)
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            print("\nAuc Score:", auc)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))

    from sklearn import linear_model

    """
    Fitting Logistic Regression: Parameters has been found using grid search
    """
    model = linear_model.LogisticRegression(C=21.544346900318832, penalty='l2', class_weight="balanced", n_jobs=-1)
    model.fit(X_train_prepared.iloc[:, :-1], X_train_prepared.iloc[:, -1])
    accuracy_lr = model.score(X_test_prepared.iloc[:, :-1], X_test_prepared.iloc[:, -1])
    print('Logistic Regression accuracy on test set of {} points: {:.4f}'.format(X_test_prepared.shape[0], accuracy_lr))

    prediction_result = pd.DataFrame()
    predictions = pd.DataFrame(model.predict_proba(X_test_prepared.iloc[:, :-1]))
    predictions.columns = ['class_' + str(item) for item in list(model.classes_)]
    predictions['true_label'] = X_test_prepared.iloc[:, -1]
    predictions['predicted_label'] = model.predict(X_test_prepared.iloc[:, :-1])
    predictions['model'] = 'LR'
    predictions['pipeline'] = 'no-pipeline'
    predictions['best_param'] = 'no-param'
    prediction_result = pd.concat([prediction_result, predictions])
    showClassificationReport(prediction_result)
    return model, new_data_prepared


import pickle

model, new_data_prepared = build_model(X_train, 1)

pickle.dump(model, open('/home/cdsw/data/model2_auc_76_11_test', 'wb'))

# Predict results on the new data
new_data_predictions = pd.DataFrame(model.predict_proba(new_data_prepared))
new_data_predictions['SK_ID_CURR'] = application_newData['SK_ID_CURR']
new_data_predictions = new_data_predictions[['SK_ID_CURR', 1]]
new_data_predictions.columns = ['SK_ID_CURR', 'TARGET']

# save new data
##############

new_data_predictions.to_csv("/home/cdsw/data/predictions_auc_76_11_test.csv", index=None)

"""
Results: ROC Test: 0.733
"""
