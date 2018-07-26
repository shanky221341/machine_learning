import os
import sys
from spark_utilities.conf import SetupEnvironment
from spark_utilities.conf import SparkConfigurations

env = SetupEnvironment()
os.system('echo "{}" | kinit'.format("Kashish70"))

# setup spark configuration
conf = SparkConfigurations()

initialExecutors = "2"
driverMemory = "8g"
maxBuffer = "1g"
driverCores = "18"
yarnQueue = "ace"
maxExecutors = "80"
crossJoin = "True"
executorMemory = "20g"
executorCores = "8"
broadCastThreshold = -1

spark = conf.spark2(initialExecutors, executorMemory, driverMemory,
                    maxBuffer, driverCores, yarnQueue, maxExecutors, crossJoin, executorCores, broadCastThreshold)

sc = spark.sparkContext
hdfs_path = "/user/vberlia/credit_prediction"

application_train = spark.read.csv(hdfs_path + "/application_train.csv")
application_train = application_train.toPandas()
sys.exit()
application_newData = spark.read.csv(hdfs_path + "/application_test.csv")
bureau = spark.read.csv(hdfs_path + "/bureau.csv")
bureau_balance = spark.read.csv(hdfs_path + "/bureau_balance.csv")
credit_card_balance = spark.read.csv(hdfs_path + "/credit_card_balance.csv")
home_credit_col_desc = spark.read.csv(hdfs_path + "/HomeCredit_columns_description.csv", encoding="ISO-8859-1")
intall_payment = spark.read.csv(hdfs_path + "/installments_payments.csv")
pos_cash = spark.read.csv(hdfs_path + "/POS_CASH_balance.csv")
prev_app = spark.read.csv(hdfs_path + "/previous_application.csv")
sample_sub = spark.read.csv(hdfs_path + "/sample_submission.csv")

print("Closing spark context")
sc.stop()
