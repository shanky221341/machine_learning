from machineLearning.missingValues import MissingValue
from machineLearning.credit_prediction import raw_features

raw_columns = raw_features.features.raw_columns


class PrepData:
    @staticmethod
    def dataPreparer1(data, has_fitted_the_main_pipelines, final_pipeline, data_type):
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
