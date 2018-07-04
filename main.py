from machineLearning.dataSummary import DataSummary
import pandas as pd

data = pd.read_csv("C:\\Users\\vberlia\\Documents\\machineLearningPackageDev\\Salary_Ranges_by_Job_Classification.csv")
print(DataSummary.returnSummaryDataFrame(data=data))
