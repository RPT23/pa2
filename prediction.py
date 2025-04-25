from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .getOrCreate()

# Load datasets with proper delimiter and quote
val_data = spark.read.option("header", "true").option("delimiter", ";").option("quote", '"').csv("s3a://rtpmodelbucket/ValidationDataset.csv")
train_data = spark.read.option("header", "true").option("delimiter", ";").option("quote", '"').csv("s3a://rtpmodelbucket/TrainingDataset.csv")

# Strip quotes from column names
val_data = val_data.toDF(*[col.strip('"') for col in val_data.columns])
train_data = train_data.toDF(*[col.strip('"') for col in train_data.columns])

# Convert all columns to float
features = [col for col in train_data.columns if col != "quality"]
for col_name in features + ["quality"]:
    train_data = train_data.withColumn(col_name, train_data[col_name].cast("float"))
    val_data = val_data.withColumn(col_name, val_data[col_name].cast("float"))

# Assemble features
assembler = VectorAssembler(inputCols=features, outputCol="features")
train_data = assembler.transform(train_data)
val_data = assembler.transform(val_data)

# Train RandomForestClassifier for multiclass
rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=50)
model = rf.fit(train_data)

# Predict
predictions = model.transform(val_data)

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy: {accuracy:.4f}")

# Stop Spark
spark.stop()
