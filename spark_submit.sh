#!/usr/bin/env bash

spark2-submit --name "predictive_modelling"  \
--py-files /home/vberlia/spark_utilities.zip,/home/vberlia/machineLearning.zip /home/vberlia/credit_risk_spark.py