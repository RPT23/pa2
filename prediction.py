from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GBTModelPrediction") \
    .getOrCreate()

# Load validation and training data from S3 with proper quote handling
val_data = spark.read.option("header", "true").option("delimiter", ";").option("quote", '"').csv("s3a://rtpmodelbucket/ValidationDataset.csv")
train_data = spark.read.option("header", "true").option("delimiter", ";").option("quote", '"').csv("s3a://rtpmodelbucket/TrainingDataset.csv")

# Clean up column names to strip excess quotes
val_data = val_data.toDF(*[col.strip('"') for col in val_data.columns])
train_data = train_data.toDF(*[col.strip('"') for col in train_data.columns])

# Cast all columns (except label) to float
features = [col for col in train_data.columns if col != "quality"]
for column in features + ["quality"]:
    train_data = train_data.withColumn(column, train_data[column].cast("float"))
    val_data = val_data.withColumn(column, val_data[column].cast("float"))

# Assemble feature vector
assembler = VectorAssembler(inputCols=features, outputCol="features")
train_data = assembler.transform(train_data)
val_data = assembler.transform(val_data)

# Train GBT classifier
gbt = GBTClassifier(labelCol="quality", featuresCol="features", maxIter=50)
model = gbt.fit(train_data)

# Make predictions
predictions = model.transform(val_data)

# Evaluate model
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy: {accuracy:.4f}")

# Stop Spark
spark.stop()
