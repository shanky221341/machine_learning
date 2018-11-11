import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from machineLearning.credit_prediction import classification_report


def generateReport(model, X_test, model_name):
    prediction_result = pd.DataFrame()
    predictions = pd.DataFrame(model.predict_proba(X_test.iloc[:, :-1]))
    predictions.columns = ['class_' + str(int(item)) for item in list(model.classes_)]
    predictions['true_label'] = X_test.iloc[:, -1]
    predictions['predicted_label'] = model.predict(X_test.iloc[:, :-1])
    predictions['model'] = model_name
    prediction_result = pd.concat([prediction_result, predictions])
    report = classification_report.ClassificationReport.showClassificationReport(prediction_result)
    return report


class PreparedData:
    def __init__(self, prep_data, strategy_name):
        self.prep_data = prep_data
        self.strategy_name = strategy_name


class Models:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name


class Stacking:
    @staticmethod
    def createStackedModel(list_prep_data, models_list, stack_model, normalize=True):
        count = 1
        all_report = pd.DataFrame()
        stack_data_layer1 = pd.DataFrame()
        stack_data_layer1_test = pd.DataFrame()
        for data_object in list_prep_data:
            application_train1 = data_object.prep_data
            application_train1 = application_train1.drop('SK_ID_CURR', axis=1)
            X = application_train1.values
            Y = application_train1['TARGET']
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
            X_train = pd.DataFrame(X_train)
            X_train.columns = application_train1.columns
            X_test = pd.DataFrame(X_test)
            X_test.columns = application_train1.columns
            if normalize:
                standardiser = StandardScaler()
                X_train_target = X_train['TARGET']
                X_test_target = X_test['TARGET']
                standardiser.fit(X_train.iloc[:, :-1])
                X_train = pd.DataFrame(standardiser.transform(X_train.iloc[:, :-1]))
                X_train['TARGET'] = X_train_target
                X_test = pd.DataFrame(standardiser.transform(X_test.iloc[:, :-1]))
                X_test['TARGET'] = X_test_target
                stack_data = pd.DataFrame()
                stack_data_test = pd.DataFrame()
            for model_object in models_list:
                model = model_object.model
                model.fit(X_train.iloc[:, :-1], X_train.iloc[:, -1])
                stack_data[model_object.model_name + '_' + data_object.strategY_name] = model.predict(
                    X_train.iloc[:, :-1])
                stack_data_test[model_object.model_name + '_' + data_object.strategY_name] = model.predict(
                    X_test.iloc[:, :-1])
                print("Showing the individual model performance: ", model_object.model_name)
                report = generateReport(model, X_test, model_object.model_name)
                all_report = pd.concat([all_report, report])
            stack_data['TARGET'] = X_train['TARGET']
            stack_data_test['TARGET'] = X_test['TARGET']
            # fitting stacking models
            model = stack_model
            model.fit(stack_data.iloc[:, :-1], stack_data.iloc[:, -1])
            stack_data_layer1['sp' + str(count)] = model.predict(
                stack_data.iloc[:, :-1])
            stack_data_layer1_test['sp' + str(count)] = model.predict(
                stack_data_test.iloc[:, :-1])
            print("Showing the individual model performance for stacking:", count)
            report = generateReport(model, stack_data_test, model_name='stacking' + count)
            all_report = pd.concat([all_report, report])
            count += 1
        stack_data_layer1['TARGET'] = stack_data['TARGET'].iloc[:, -1]
        stack_data_layer1_test['TARGET'] = stack_data_test['TARGET'].iloc[:, -1]
        # fitting final stacked model

        model = stack_model
        model.fit(stack_data_layer1.iloc[:, :-1], stack_data_layer1.iloc[:, -1])
        print("Showing final stacking performance")
        report = generateReport(model, stack_data_layer1_test, model_name='final_stacking')
        all_report = pd.concat([all_report, report])
        return all_report


# list_prep_data = PreparedData(prep_data=
