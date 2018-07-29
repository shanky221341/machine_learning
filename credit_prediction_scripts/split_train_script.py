import pandas as pd

"""
This script is to split
the training set into six almost equal parts
Last set contains slightly more rows.

"""

application_train = pd.read_csv("/home/cdsw/data/application_train.csv")

train_sample1 = application_train.sample(50000, random_state=0)

remaining_indices = list(set(application_train.index) - set(train_sample1.index))

train_sample2 = application_train.ix[remaining_indices].sample(50000, random_state=0)

remaining_indices = list(set(application_train.index) - (set(train_sample1.index) | set(train_sample2.index)))

train_sample3 = application_train.ix[remaining_indices].sample(50000, random_state=0)

remaining_indices = list(set(application_train.index) - (set(train_sample1.index) | set(train_sample2.index)
                                                         | set(train_sample3.index)))

train_sample4 = application_train.ix[remaining_indices].sample(50000, random_state=0)

remaining_indices = list(set(application_train.index) - (set(train_sample1.index) | set(train_sample2.index)
                                                         | set(train_sample3.index) | set(train_sample4.index)))

train_sample5 = application_train.ix[remaining_indices].sample(50000, random_state=0)

remaining_indices = list(set(application_train.index) - (set(train_sample1.index) | set(train_sample2.index)
                                                         | set(train_sample3.index) |
                                                         set(train_sample4.index) | set(train_sample5.index)))

train_sample6 = application_train.ix[remaining_indices]

train_sample1.to_csv("/home/cdsw/data/train_sample1.csv", index=None)
train_sample2.to_csv("/home/cdsw/data/train_sample2.csv", index=None)
train_sample3.to_csv("/home/cdsw/data/train_sample3.csv", index=None)
train_sample4.to_csv("/home/cdsw/data/train_sample4.csv", index=None)
train_sample5.to_csv("/home/cdsw/data/train_sample5.csv", index=None)
train_sample6.to_csv("/home/cdsw/data/train_sample6.csv", index=None)
