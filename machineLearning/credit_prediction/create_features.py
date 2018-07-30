from machineLearning.featureCreation import IsMissingFeature
from machineLearning.missingValues import CustomQuantitativeImputer
from machineLearning.credit_prediction import raw_features
from machineLearning.featureCreation import CreateFrequencyLookupFeature
from machineLearning.featureCreation import CreateOneHotEncoding
from machineLearning.featureCreation import ClipOutliers
from sklearn.pipeline import Pipeline

miss_columns = raw_features.features.miss_columns
raw_columns = raw_features.features.raw_columns
freq_columns = raw_features.features.freq_columns
one_hot_columns = raw_features.features.one_hot_columns


class CreateFeatures:
    @staticmethod
    def create_all_features1():
        # all_pipes will contain pipeline of different feature categories
        all_pipes = []

        # is missing feature
        is_missing_pipes = []
        is_missing_pipes.append(('is_miss_features', IsMissingFeature(cols=miss_columns)))

        # Imputing Missing values for numerical columns using "mean"
        quant_miss_pipes = []
        quant_miss_pipes.append(('miss_impute_cols', CustomQuantitativeImputer(cols=raw_columns, strategy="median")))

        # outlier_pipes = []
        # outlier_pipes.append(
        #     ('rem_otlr_' + 'YEARS_BEGINEXPLUATATION_MEDI', ClipOutliers('YEARS_BEGINEXPLUATATION_MEDI', 10, 90)))
        # outlier_pipes.append(('rem_otlr_' + 'LANDAREA_AVG', ClipOutliers('LANDAREA_AVG', 1, 98)))
        # outlier_pipes.append(('rem_otlr_' + 'LIVINGAPARTMENTS_MODE', ClipOutliers('LIVINGAPARTMENTS_MODE', 1, 98)))

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
        # all_pipes.append(Pipeline(outlier_pipes))

        all_pipelines = []
        count = 1
        for pipe in all_pipes:
            all_pipelines.append(('feature_set' + str(count), pipe))
            count += 1

        return Pipeline(all_pipelines)
