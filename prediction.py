from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# Initialize Spark session
spark = SparkSession.builder.appName("GBTModelPrediction").getOrCreate()

# Read datasets from S3
train_data = spark.read.csv("s3a://rtpmodelbucket/TrainingDataset.csv", header=True, inferSchema=True)
val_data = spark.read.csv("s3a://rtpmodelbucket/ValidationDataset.csv", header=True, inferSchema=True)

# Clean column names
train_data = train_data.toDF(*[col.replace('"', '').strip() for col in train_data.columns])
val_data = val_data.toDF(*[col.replace('"', '').strip() for col in val_data.columns])

# Filter binary classification only (0 and 1 labels)
train_data = train_data.filter((train_data["quality"] == 0) | (train_data["quality"] == 1))
val_data = val_data.filter((val_data["quality"] == 0) | (val_data["quality"] == 1))

# Feature columns
feature_cols = [col for col in train_data.columns if col != "quality"]

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data = assembler.transform(train_data)
val_data = assembler.transform(val_data)

# Train GBT Classifier
gbt = GBTClassifier(labelCol="quality", featuresCol="features", maxIter=50)
model = gbt.fit(train_data)

# Make predictions
predictions = model.transform(val_data)

# Evaluate Accuracy
evaluator_acc = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)

# Evaluate F1 Score
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Save results to local text file
with open("prediction_results.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1_score:.4f}\n")

# Save model to S3
model.save("s3a://rtpmodelbucket/wine_quality_model")
