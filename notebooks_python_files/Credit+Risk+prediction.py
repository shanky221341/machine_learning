
# coding: utf-8

# In[1]:

import pandas as pd
import sys
import matplotlib.pyplot as plt
pd.options.display.max_columns=1000
pd.set_option('display.max_colwidth', -1)
sys.path.insert(0, "C:\\Users\\vberlia\\Documents\\machine_learning")


# In[2]:

from machineLearning.dataSummary import DataSummary
from machineLearning.visualizations import Visualization
from machineLearning.missingValues import MissingValue
from machineLearning.models import Model
from machineLearning.modelInputs import KNNInputs
from machineLearning.pipelines import Pipelines
from machineLearning.featureCreation import CreateMeanLookupFeature
from machineLearning.featureCreation import CreateMedianLookupFeature
from machineLearning.featureCreation import CreateFrequencyLookupFeature
from machineLearning.featureCreation import CreateOneHotEncoding
from machineLearning.featureCreation import CustomCutter
from machineLearning.misc import Misc
from machineLearning.missingValues import CustomEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import ggplot
from ggplot import *
import numpy as np


# In[3]:

application_train=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/application_train.csv")
# application_newData=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/application_test.csv")
# bureau=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/bureau.csv")
# bureau_balance=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/bureau_balance.csv")
# # credit_card_balance=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/credit_card_balance.csv")
home_credit_col_desc=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/HomeCredit_columns_description.csv",encoding = "ISO-8859-1")
# intall_payment=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/installments_payments.csv")
# pos_cash=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/POS_CASH_balance.csv")
# prev_app=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/previous_application.csv")
# sample_submi=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/sample_submission.csv")


# ### Data Insights- Gathering maximum knowledge about the data
# 1. Total number of training rows 307511
# 2. Annuity amount is highly skewed data
# 3. Annuity amount with class '1' does not contains any missing data but class '0' contains 12 missing values
# 4. Its highly unbalanced classification problem-> around 92% people did not have payment difficulties
# and only 8% had late payment issues.
# 5. Need to find the characteristics which actually separates them.
# 6. Bureau data frame contains the information about the previous loan history of people who applied for loan
# from different financial institition. Lot of features can be generated from this.
# 7.

# In[4]:

home_credit_col_desc


# In[5]:

approved_sk_ids=application_train[application_train['TARGET']==1]['SK_ID_CURR']
rejected_sk_ids=application_train[application_train['TARGET']==0]['SK_ID_CURR']
# len(rejected_sk_ids)
set(rejected_sk_ids) & set(bureau['SK_ID_CURR'])
# set(application_train['SK_ID_CURR'])-set(bureau['SK_ID_CURR'])


# ### Preparing Features

# ### Since whole data is not able to fit in memory, working on sample data

# In[5]:

app_sum=DataSummary.returnSummaryDataFrame(application_train)


# In[93]:

DataSummary.returnFrequencyCounts(columns=['class'],data=X_train_prepared,normalize=True)


# In[ ]:

# in the gender column


# In[4]:

train_sample=application_train.sample(10000)


# In[5]:

X=train_sample.values
Y=train_sample['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


# In[6]:

X_train=pd.DataFrame(X_train)
X_train.columns=train_sample.columns
X_test=pd.DataFrame(X_test)
X_test.columns=train_sample.columns


# In[7]:

freq_columns=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN']
one_hot_columns=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN_freq_binned']

# This bins and the corresponding labels you will have to create manually after observing the range of the 
# numeric values, so that the bins covers all the values.
bin_columns={'CNT_CHILDREN_freq':[[0, 100, 1500,4000,6500],['high_num_child', 'medium_num_child', 'low_num_child','no_child']]}

global has_fitted_the_main_pipelines
has_fitted_the_main_pipelines=0
def create_all_features(freq_columns):
    # all_pipes will contain pipeline of different feature categories
    all_pipes=[]

    # Creating freqeuncy feature pipelines
    freq_pipes=[]
    for col in freq_columns:
        freq_pipes.append(('freq_by_'+col,CreateFrequencyLookupFeature(categorical_column=col,new_col_name=
                                         col+'_freq')))

    # Creating binning features
    bin_pipes=[]
    for col in bin_columns.keys():
        bin_pipes.append(('binned_'+col,CustomCutter(col=col,bins=bin_columns[col][0],labels=bin_columns[col][1])))

    # Creating one-hot feature pipelines
    one_hot_pipes=[]
    for col in one_hot_columns:
            one_hot_pipes.append(('one_hot_'+col,CreateOneHotEncoding(categorical_column=col)))

    # Create high level pipeline for all feature categories         
    all_pipes.append(Pipeline(freq_pipes))
    all_pipes.append(Pipeline(bin_pipes))
    all_pipes.append(Pipeline(one_hot_pipes))
    
    all_pipelines=[]
    count=1
    for pipe in all_pipes:
        all_pipelines.append(('feature_set'+str(count),pipe))
        count+=1

    return Pipeline(all_pipelines)   
final_pipeline=create_all_features(freq_columns)


# In[8]:

"""
Creating adhoc features
"""

# Checking the counts of contract type across loan issues class(0,1)
contract_type_vs_loan_issues = pd.crosstab(index=X_train["NAME_CONTRACT_TYPE"], 
                           columns=X_train["TARGET"])

total_no_loan_issues=np.sum(contract_type_vs_loan_issues[0])
total_loan_issues=np.sum(contract_type_vs_loan_issues[1])
# Creating the lookup for prevalance of cash loans and revolving loans across loan issues and no loan issues

tmp=pd.DataFrame(contract_type_vs_loan_issues[0]).transpose().to_dict()

prev_no_loan_issues={}
for key in tmp.keys():
    prev_no_loan_issues[key]=np.round(tmp[key][0]/total_no_loan_issues,3)

tmp=pd.DataFrame(contract_type_vs_loan_issues[1]).transpose().to_dict()
prev_loan_issues={}
for key in tmp.keys():
    prev_loan_issues[key]=np.round(tmp[key][1]/total_loan_issues,3)    

# Final feature set columns    
columns=['prev_no_loan_issue','prev_loan_issue','NAME_CONTRACT_TYPE_freq','name_contract_type_cash_loans', 'name_contract_type_revolving_loans']


# In[9]:

# Data Preparer function -> should prepare data for all train, test and new data. Because
# every data kind(train, test and new data) goes through the same preparation 
# phase
def dataPreparer(data,columns,has_fitted_the_main_pipelines,final_pipeline):
    newData=data.copy()
    colsA=newData.columns

    # It is observed that there are few vals with 'XNA' code and replacing them with 'F'
    newData=MissingValue.replaceValuesInColumns(columns=['CODE_GENDER'],data=newData,replace_with_val='F',val_to_replace='XNA')
    
    # Adding two new features
    newData['prev_no_loan_issue']=newData['NAME_CONTRACT_TYPE'].map(lambda x: prev_no_loan_issues[x])
    newData['prev_loan_issue']=newData['NAME_CONTRACT_TYPE'].map(lambda x: prev_loan_issues[x])

    if has_fitted_the_main_pipelines==0:
        final_pipeline=final_pipeline.fit(newData)
        has_fitted_the_main_pipelines=1
    newData=final_pipeline.transform(newData)
    colsB=newData.columns
    columns=list(colsB[len(colsA):len(colsB)])    
    columns.append('CNT_CHILDREN')
    columns.append('TARGET')
    return newData[columns],has_fitted_the_main_pipelines


# In[25]:

X_train.head(4)


# In[10]:

'''
This should be underlying order
'''
X_train_prepared,has_fitted_the_main_pipelines=dataPreparer(X_train,columns,has_fitted_the_main_pipelines,final_pipeline)
# X_test_prepared,has_fitted_the_main_pipelines=dataPreparer(X_test,columns,has_fitted_the_main_pipelines,final_pipeline)
# new_data_prepared,has_fitted_the_main_pipelines=dataPreparer(new_data,columns,has_fitted_the_main_pipelines,final_pipeline)


# In[11]:

# np.unique(X_train_prepared['TARGET'],return_counts=True)
Misc.rename_columns(columns={'TARGET':'class'},data=X_train_prepared)
X_train_prepared=Misc.drop_columns(data=X_train_prepared,columns=['CNT_CHILDREN_freq_binned'])


# In[12]:

X_train_prepared.dtypes


# In[77]:

X_train_prepared.head(3)


# In[70]:

X_train_prepared.head(3)


# In[57]:

X_test_prepared.head(3)


# In[ ]:

X_test_prepared.head(100)


# In[ ]:

Pipelines.pipelines


# In[67]:

X_train.head(3)


# In[83]:

set(X_train_prepared['class'])


# In[13]:

X_train_prepared['class'] = X_train_prepared['class'].astype('int32')
X_train_prepared['CNT_CHILDREN'] = X_train_prepared['CNT_CHILDREN'].astype('int32')


# In[14]:

X_train_prepared.dtypes


# In[15]:

result=Model.fit_knn_pipelines(X_train_prepared,Pipelines.pipelines,5)


# In[94]:

DataSummary.returnFrequencyCounts(columns=['class'],data=X_train_prepared,normalize=True)


# In[96]:

result


# In[94]:

final_pipeline.__dict__['steps'][0][1].__dict__['steps'][0][1].__dict__


# In[71]:

pd.DataFrame(list(np.unique(X_train['NAME_CONTRACT_TYPE'],return_counts=True)))


# In[83]:

X_train['TARGET'] = X_train.TARGET.astype(float)


# In[86]:

dict(X_train.groupby(['NAME_CONTRACT_TYPE'])['NAME_CONTRACT_TYPE'].count())


# In[ ]:

X_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].agg(['mean'])


# In[107]:

keys=list(set(X_train['NAME_CONTRACT_TYPE']))


# In[108]:

dictionary = dict(zip(keys, values))


# In[105]:

values=list(range(0,len(a)))


# In[112]:

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()


# In[177]:

cc=CreateOneHotEncoding(categorical_column='NAME_CONTRACT_TYPE')


# In[174]:

['NAME_CONTRACT_TYPE'.lower()+'_'+item.lower().replace(" ","_") for  item in list(cc.mlb.classes_)]


# In[173]:

'NAME_CONTRACT_TYPE'.lower()+'_'


# In[157]:

X_train_prepared.head(2).join(cc.transform(X_train).head(2))


# In[178]:

cc.fit(X_train)


# In[180]:

cc.transform(X_train_prepared)


# In[131]:

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

pd.DataFrame(mlb.fit_transform(X_train['NAME_CONTRACT_TYPE'].map(lambda x:[x])),columns=mlb.classes_)


# 

# In[127]:

X_train['NAME_CONTRACT_TYPE'].head()


# In[128]:

X_train['NAME_CONTRACT_TYPE'].map(lambda x:[x])


# In[116]:

ohe.fit_transform(a).values.reshape(-1,1)


# In[113]:

a=X_train['NAME_CONTRACT_TYPE'].map(lambda x:dictionary[x]).head()


# In[96]:

pd.get_dummies(X_train['NAME_CONTRACT_TYPE'])


# In[ ]:

DataSummary.returnFrequencyCounts(columns=['TARGET','NAME_CONTRACT_TYPE'],data=X_train)


# In[20]:

sns.countplot(x="TARGET", hue="NAME_CONTRACT_TYPE", data=application_train)
plt.show()


# In[6]:

app_train_sum=DataSummary.returnSummaryDataFrame(application_train)


# In[17]:

import seaborn as sns
sns.categorical(x="TARGET", y="AMT_ANNUITY", kind="box", data=application_train);


# In[14]:

application_train["AMT_ANNUITY"].plot(kind="density",  # Create density plot
                      figsize=(8,8),    # Set figure size
                      xlim= (0,5))
plt.show()


# In[12]:

application_train.boxplot(column="AMT_ANNUITY", by= "TARGET")
plt.show()


# In[7]:

app_train_sum['NAME_CONTRACT_TYPE']


# In[49]:

np.nanmedian(application_train[application_train['TARGET'] == 0][['AMT_ANNUITY']])


# In[54]:

annuity_median_class_0=np.nanmedian(application_train[application_train['TARGET'] == 0][['AMT_ANNUITY']])
annuity_mean_class_0=np.nanmean(application_train[application_train['TARGET'] == 0][['AMT_ANNUITY']])
tmp=MissingValue.replaceValuesInColumns(data=application_train, val_to_replace=None, replace_with_val=annuity_median_class_0, columns=['AMT_ANNUITY'])


# In[56]:

Visualization.createHistPlotForVarsForBinaryClass(data=tmp, label_column='TARGET', columns=['AMT_ANNUITY'], zero_meaning='Loan Rejected', one_meaning='Loan Approved')


# In[12]:

home_credit_col_desc=pd.read_csv("C:/Users/vberlia/Documents/data/credit_prediction/all/HomeCredit_columns_description.csv",encoding = "ISO-8859-1")


# In[13]:

home_credit_col_desc


# In[ ]:

DataSummary.returnSummaryDataFrame()

