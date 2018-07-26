from pyspark.sql import HiveContext
from  pyspark.sql import SparkSession
from pyspark import SparkConf
import os
import sys


class SetupEnvironment:
    def __init__(self):
        """in production kerberos authentication will be done using file and setting up environment will not be required"""
        os.environ["SPARK_HOME"] = "/opt/cloudera/parcels/SPARK2/lib/spark2"
        os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
        os.environ["PYSPARK_PYTHON"] = "/opt/cloudera/parcels/Anaconda/bin/python"
        sys.path.insert(0, os.environ["PYLIB"] + "/py4j-0.10.4-src.zip")
        sys.path.insert(0, os.environ["PYLIB"] + "/pyspark.zip")


class SparkConfigurations:
    params = {"IE": "spark.dynamicAllocation.initialExecutors",
              "DM": "spark.driver.memory",
              "MB": "spark.kryoserializer.buffer.max",
              "DC": "spark.driver.cores",
              "YQ": "spark.yarn.queue",
              "ME": "spark.dynamicAllocation.maxExecutors",
              "CR": "spark.sql.crossJoin.enabled",
              "EM": "spark.executor.memory",
              "EC": "spark.executor.cores",
              "BT": "spark.sql.autoBroadcastJoinThreshold"
              }

    def spark1_6_1(self, initialExecutors, executorMemory, driverMemory, maxBuffer, driverCores, yarnQueue,
                   maxExecutors, crossJoin, executorCores):
        from pyspark import SparkConf
        from pyspark import SparkContext
        conf = SparkConf().set(SparkConfigurations.params["IE"], initialExecutors) \
            .set(SparkConfigurations.params['DM'], driverMemory) \
            .set(SparkConfigurations.params['MB'], maxBuffer) \
            .set(SparkConfigurations.params['DC'], driverCores) \
            .set(SparkConfigurations.params['YQ'], yarnQueue) \
            .set(SparkConfigurations.params['ME'], maxExecutors) \
            .set(SparkConfigurations.params['CR'], crossJoin) \
            .set(SparkConfigurations.params['EM'], executorMemory) \
            .set(SparkConfigurations.params['EC'], executorCores)

        sc = SparkContext(conf=conf)
        hc = HiveContext(sc)
        return sc, hc

    def spark2(self, initialExecutors, executorMemory, driverMemory, maxBuffer, driverCores, yarnQueue, maxExecutors,
               crossJoin, executorCores, broadCastThreshold):
        conf = SparkConf().set(SparkConfigurations.params["IE"], initialExecutors) \
            .set(SparkConfigurations.params['DM'], driverMemory) \
            .set(SparkConfigurations.params['MB'], maxBuffer) \
            .set(SparkConfigurations.params['DC'], driverCores) \
            .set(SparkConfigurations.params['YQ'], yarnQueue) \
            .set(SparkConfigurations.params['ME'], maxExecutors) \
            .set(SparkConfigurations.params['CR'], crossJoin) \
            .set(SparkConfigurations.params['EM'], executorMemory) \
            .set(SparkConfigurations.params['EC'], executorCores) \
            .set(SparkConfigurations.params['BT'], broadCastThreshold)

        spark = SparkSession \
            .builder \
            .config(conf=conf) \
            .enableHiveSupport() \
            .getOrCreate()
        return spark
