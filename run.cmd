:: Navigate to where you have putty installed

C:\Users\vberlia\AppData\Local\Continuum\Anaconda3\python.exe zipModules.py "spark_utilities"
C:\Users\vberlia\AppData\Local\Continuum\Anaconda3\python.exe zipModules.py "machineLearning"

cd C:\Program Files (x86)\Putty\

pscp.exe C:\Users\vberlia\Documents\machine_learning\spark_utilities.zip vberlia@cpks99hdge01r:/home/vberlia
pscp.exe C:\Users\vberlia\Documents\machine_learning\machineLearning.zip vberlia@cpks99hdge01r:/home/vberlia
pscp.exe C:\Users\vberlia\Documents\machine_learning\credit_risk_spark.py vberlia@cpks99hdge01r:/home/vberlia

plink.exe -ssh vberlia@cpks99hdge01r -pw Kashish70 -m C:\Users\vberlia\Documents\machine_learning\spark_submit.sh > C:\Users\vberlia\Documents\machine_learning\credit_risk_logs\log1.txt