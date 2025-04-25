from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ✅ Start Spark session
spark = SparkSession.builder \
    .appName("GBTModelPrediction") \
    .getOrCreate()

# ✅ Load the validation data
val_data = spark.read.csv("s3a://rtpmodelbucket/ValidationDataset.csv", header=True, inferSchema=True)

# ✅ Drop rows with missing or empty values
val_data = val_data.na.drop()

# ✅ List of feature columns (replace with your actual training feature columns)
feature_cols = [col for col in val_data.columns if col != 'label']

# ✅ Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
val_data = assembler.transform(val_data)

# ✅ Load the trained model
model = PipelineModel.load("s3a://rtpmodelbucket/trainingmodel.model")

# ✅ Make predictions
predictions = model.transform(val_data)

# ✅ Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy = {accuracy:.4f}")
