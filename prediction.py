from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("GBTModelPrediction").getOrCreate()

# Load datasets
train_data = spark.read.csv("s3a://rtpmodelbucket/TrainingDataset.csv", header=True, inferSchema=True)
val_data = spark.read.csv("s3a://rtpmodelbucket/ValidationDataset.csv", header=True, inferSchema=True)

# Clean column names
train_data = train_data.toDF(*[col.strip('"') for col in train_data.columns])
val_data = val_data.toDF(*[col.strip('"') for col in val_data.columns])

# Cast columns to float
for column in train_data.columns:
    train_data = train_data.withColumn(column, train_data[column].cast("float"))
    val_data = val_data.withColumn(column, val_data[column].cast("float"))

# Filter for binary classification: quality == 0 or 1
train_data = train_data.filter((train_data["quality"] == 0) | (train_data["quality"] == 1))
val_data = val_data.filter((val_data["quality"] == 0) | (val_data["quality"] == 1))

# Assemble features
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data = assembler.transform(train_data)
val_data = assembler.transform(val_data)

# Train GBT model
gbt = GBTClassifier(labelCol="quality", featuresCol="features", maxIter=50)
model = gbt.fit(train_data)

# Make predictions
predictions = model.transform(val_data)

# Evaluate using Accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy"
)
accuracy = accuracy_evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Evaluate using F1 Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="f1"
)
f1_score = f1_evaluator.evaluate(predictions)
print(f"Test F1 Score: {f1_score:.4f}")

spark.stop()
