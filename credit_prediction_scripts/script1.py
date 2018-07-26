import pandas as pd
pd.options.display.max_columns = 1000
pd.set_option('display.max_colwidth', -1)
# sys.path.insert(0, "C:\\Users\\vberlia\\Documents\\machine_learning")

from machineLearning.missingValues import MissingValue
from machineLearning.featureCreation import CreateFrequencyLookupFeature
from machineLearning.featureCreation import CreateOneHotEncoding
from machineLearning.featureCreation import CustomCutter
from machineLearning.misc import Misc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

application_train = pd.read_csv("/home/cdsw/data/application_train.csv")
application_newData = pd.read_csv("/home/cdsw/data/application_test.csv")
home_credit_col_desc = pd.read_csv("/home/cdsw/data/HomeCredit_columns_description.csv", encoding="ISO-8859-1")
sample_submi = pd.read_csv("/home/cdsw/data/sample_submission.csv")

X = application_train.values
Y = application_train['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

X_train = pd.DataFrame(X_train)
X_train.columns = application_train.columns
X_test = pd.DataFrame(X_test)
X_test.columns = application_train.columns

freq_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN']
one_hot_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN_freq_binned']

# This bins and the corresponding labels you will have to create manually after observing the range of the
# numeric values, so that the bins covers all the values.
bin_columns = {'CNT_CHILDREN_freq': [[0, 100, 500, 4000, 60000, 200000],
                                     ['very_high_num_child', 'high_num_child', 'medium_num_child', 'low_num_child',
                                      'no_child']]}

global has_fitted_the_main_pipelines
has_fitted_the_main_pipelines = 0


def create_all_features(freq_columns):
    # all_pipes will contain pipeline of different feature categories
    all_pipes = []

    # Creating freqeuncy feature pipelines
    freq_pipes = []
    for col in freq_columns:
        freq_pipes.append(('freq_by_' + col, CreateFrequencyLookupFeature(categorical_column=col, new_col_name=
        col + '_freq')))

    # Creating binning features
    bin_pipes = []
    for col in bin_columns.keys():
        bin_pipes.append(('binned_' + col, CustomCutter(col=col, bins=bin_columns[col][0], labels=bin_columns[col][1])))

    # Creating one-hot feature pipelines
    one_hot_pipes = []
    for col in one_hot_columns:
        one_hot_pipes.append(('one_hot_' + col, CreateOneHotEncoding(categorical_column=col)))

    # Create high level pipeline for all feature categories
    all_pipes.append(Pipeline(freq_pipes))
    all_pipes.append(Pipeline(bin_pipes))
    all_pipes.append(Pipeline(one_hot_pipes))

    all_pipelines = []
    count = 1
    for pipe in all_pipes:
        all_pipelines.append(('feature_set' + str(count), pipe))
        count += 1

    return Pipeline(all_pipelines)


final_pipeline = create_all_features(freq_columns)

"""
Creating adhoc features
"""

# Checking the counts of contract type across loan issues class(0,1)
contract_type_vs_loan_issues = pd.crosstab(index=X_train["NAME_CONTRACT_TYPE"],
                                           columns=X_train["TARGET"])

total_no_loan_issues = np.sum(contract_type_vs_loan_issues[0])
total_loan_issues = np.sum(contract_type_vs_loan_issues[1])
# Creating the lookup for prevalance of cash loans and revolving loans across loan issues and no loan issues

tmp = pd.DataFrame(contract_type_vs_loan_issues[0]).transpose().to_dict()

prev_no_loan_issues = {}
for key in tmp.keys():
    prev_no_loan_issues[key] = np.round(np.float(tmp[key][0]) / total_no_loan_issues, 3)

tmp = pd.DataFrame(contract_type_vs_loan_issues[1]).transpose().to_dict()
prev_loan_issues = {}
for key in tmp.keys():
    prev_loan_issues[key] = np.round(np.float(tmp[key][1]) / total_loan_issues, 3)

# Final feature set columns
columns = ['prev_no_loan_issue', 'prev_loan_issue', 'NAME_CONTRACT_TYPE_freq', 'name_contract_type_cash_loans',
           'name_contract_type_revolving_loans']


# In[9]:

# Data Preparer function -> should prepare data for all train, test and new data. Because
# every data kind(train, test and new data) goes through the same preparation
# phase
def dataPreparer(data, columns, has_fitted_the_main_pipelines, final_pipeline, data_type):
    newData = data.copy()
    colsA = newData.columns

    # It is observed that there are few vals with 'XNA' code and replacing them with 'F'
    newData = MissingValue.replaceValuesInColumns(columns=['CODE_GENDER'], data=newData, replace_with_val='F',
                                                  val_to_replace='XNA')

    # Adding two new features
    newData['prev_no_loan_issue'] = newData['NAME_CONTRACT_TYPE'].map(lambda x: prev_no_loan_issues[x])
    newData['prev_loan_issue'] = newData['NAME_CONTRACT_TYPE'].map(lambda x: prev_loan_issues[x])

    if has_fitted_the_main_pipelines == 0:
        final_pipeline = final_pipeline.fit(newData)
        has_fitted_the_main_pipelines = 1
    newData = final_pipeline.transform(newData)
    colsB = newData.columns
    columns = list(colsB[len(colsA):len(colsB)])
    columns.append('CNT_CHILDREN')
    if data_type != 'new_data':
        columns.append('TARGET')
    return newData[columns], has_fitted_the_main_pipelines


'''
This should be underlying order
'''
X_train_prepared, has_fitted_the_main_pipelines = dataPreparer(X_train, columns, has_fitted_the_main_pipelines,
                                                               final_pipeline, 'train')
X_test_prepared, has_fitted_the_main_pipelines = dataPreparer(X_test, columns, has_fitted_the_main_pipelines,
                                                              final_pipeline, 'test')
new_data_prepared, has_fitted_the_main_pipelines = dataPreparer(application_newData, columns,
                                                                has_fitted_the_main_pipelines, final_pipeline,
                                                                'new_data')

Misc.rename_columns(columns={'TARGET': 'class'}, data=X_train_prepared)
X_train_prepared = Misc.drop_columns(data=X_train_prepared, columns=['CNT_CHILDREN_freq_binned'])
X_train_prepared['class'] = X_train_prepared['class'].astype('int32')
X_train_prepared['CNT_CHILDREN'] = X_train_prepared['CNT_CHILDREN'].astype('int32')

Misc.rename_columns(columns={'TARGET': 'class'}, data=X_test_prepared)
X_test_prepared = Misc.drop_columns(data=X_test_prepared, columns=['CNT_CHILDREN_freq_binned'])
X_test_prepared['class'] = X_test_prepared['class'].astype('int32')
X_test_prepared['CNT_CHILDREN'] = X_test_prepared['CNT_CHILDREN'].astype('int32')

new_data_prepared = Misc.drop_columns(data=new_data_prepared, columns=['CNT_CHILDREN_freq_binned'])
new_data_prepared['CNT_CHILDREN'] = new_data_prepared['CNT_CHILDREN'].astype('int32')

# Saving Datasets
X_train_prepared.to_csv("/home/cdsw/data/x_train_prepared1.csv", index=None)
X_test_prepared.to_csv("/home/cdsw/data/x_test_prepared1.csv", index=None)
new_data_prepared.to_csv("/home/cdsw/data/new_data_pepared1.csv", index=None)


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
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

from sklearn import linear_model

"""
Fitting Logistic Regression: Parameters has been found using grid search
"""
model = linear_model.LogisticRegression(C=21.544346900318832, penalty='l2', class_weight="balanced")
model.fit(X_train_prepared.iloc[:, :-1], X_train_prepared.iloc[:, -1])
accuracy_lr = model.score(X_test_prepared.iloc[:, :-1], X_test_prepared.iloc[:, -1])
print('Logistic Regression accuracy on test set of {} points: {:.4f}'.format(X_test_prepared.shape[0], accuracy_lr))

import pickle

pickle.dump(model, open('/home/cdsw/data/model1.csv', 'wb'))

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

# Predict results on the new data
new_data_predictions = pd.DataFrame(model.predict_proba(new_data_prepared))
new_data_predictions['SK_ID_CURR'] = application_newData['SK_ID_CURR']
new_data_predictions = new_data_predictions[['SK_ID_CURR', 1]]
new_data_predictions.columns = ['SK_ID_CURR', 'TARGET']

# save new data
##############

new_data_predictions.to_csv("/home/cdsw/data/predictions1.csv", index=None)


"""
Results: ROC Test: 0.578
"""