from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("GBTModelPrediction").getOrCreate()

# Read the validation data from S3 with correct delimiter and quote
val_data = spark.read \
    .option("header", "true") \
    .option("delimiter", ";") \
    .option("quote", '"') \
    .csv("s3a://rtpmodelbucket/ValidationDataset.csv")

# Read the training data from S3 with correct delimiter and quote
train_data = spark.read \
    .option("header", "true") \
    .option("delimiter", ";") \
    .option("quote", '"') \
    .csv("s3a://rtpmodelbucket/TrainingDataset.csv")

# Convert columns to correct types (ensure all features are float and label is int)
for column in train_data.columns:
    if column != "quality":
        train_data = train_data.withColumn(column, train_data[column].cast("float"))
        val_data = val_data.withColumn(column, val_data[column].cast("float"))
train_data = train_data.withColumn("quality", train_data["quality"].cast("int"))
val_data = val_data.withColumn("quality", val_data["quality"].cast("int"))

# Assemble features into a feature vector
feature_columns = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
val_data = assembler.transform(val_data)

# Load the saved model from S3
model_path = "s3a://rtpmodelbucket/trainingmodel.model"
model = GBTClassificationModel.load(model_path)

# Make predictions on validation data
predictions = model.transform(val_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy = {accuracy:.4f}")

# Stop the Spark session
spark.stop()
