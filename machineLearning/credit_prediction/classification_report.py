from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from machineLearning.misc import Misc


class ClassificationReport:
    @staticmethod
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
            print("\nParam ->", set(prediction_result[prediction_result['pipeline'] == pipeline]['param']))
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

            report = Misc.build_precision_table_ver2(classification_report(y_true, y_pred))
            report.insert(0, 'AUC', auc)
            return report
