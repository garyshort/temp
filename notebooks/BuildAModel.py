# Databricks notebook source
# MAGIC %md ##Authenticate against the blog store

# COMMAND ----------

# Authenticate
spark.conf.set(
  "fs.azure.account.key.vodafonedemoblob.blob.core.windows.net",
  "gReyIPdEV1hhCIOfJgGa9h1H6p9fddpnxUf6GscgVpemV/SC7m8KFXca61x4MV9V2eyxUmHWL76ti6F3zWlaNA==")



# COMMAND ----------

# Clean the data
df = spark.read.csv("wasbs://analytics@vodafonedemoblob.blob.core.windows.net/ChurnData.csv", header=True)
display(df)


# COMMAND ----------

from pyspark.sql.types import IntegerType

df = df.withColumn("NumberOfSupportCalls", df["NumberOfSupportCalls"].cast(IntegerType()))
df = df.withColumn("FailedConnections", df["FailedConnections"].cast(IntegerType()))

df = df.drop("Email")
display(df)
df.dtypes

# COMMAND ----------

# Build a model

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

stages = []
cols = df.columns

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="Churn", outputCol="label")
stages += [label_stringIdx]

# Transform all features into a vector using VectorAssembler
assembler = VectorAssembler(inputCols=["NumberOfSupportCalls", "FailedConnections"], outputCol="features")
stages += [assembler]

# Create a Pipeline.
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
dataset = pipelineModel.transform(df)

# Keep relevant columns
selectedcols = ["label", "features"] + cols
df = dataset.select(selectedcols)
display(df)

# COMMAND ----------

# Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "NumberOfSupportCalls", "FailedConnections")
display(selected)

# COMMAND ----------

# Persist the model
import os

model_name = "ChurnModel.mml"
model_dbfs = os.path.join("/dbfs", model_name)

lrModel.write().overwrite().save(model_name)
print("saved model to {}".format(model_dbfs))


# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls -la /dbfs/ChurnModel.mml/*