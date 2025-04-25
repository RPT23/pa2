import findspark
findspark.init()
findspark.find()

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
import sys

def prepare_input(df):
    # Clean column names
    renamed_cols = [col.replace('"', '') for col in df.columns]
    df = df.toDF(*renamed_cols)

    # Index target column
    label_indexer = StringIndexer(inputCol="quality", outputCol="label", handleInvalid="keep")
    df = label_indexer.fit(df).transform(df)

    # Assemble feature columns
    features = [col for col in df.columns if col not in ("quality", "label")]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    df = assembler.transform(df)

    return df

def run_prediction(test_csv, model_dir):
    try:
        spark = SparkSession.builder \
            .appName("GBTModelPrediction") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .getOrCreate()

        bucket = "rtpmodelbucket"
        test_path = f"s3a://{bucket}/{test_csv}"
        model_path = f"s3a://{bucket}/{model_dir}"

        # Load and prepare data
        raw_test = spark.read.csv(test_path, header=True, inferSchema=True, sep=";")
        prepared_test = prepare_input(raw_test)

        # Load model
        model = PipelineModel.load(model_path)

        # Make predictions
        results = model.transform(prepared_test)

        # Evaluate
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction"
        )

        accuracy = evaluator.evaluate(results, {evaluator.metricName: "accuracy"})
        f1 = evaluator.evaluate(results, {evaluator.metricName: "f1"})

        # Round metrics for display
        accuracy = round(accuracy, 4)
        f1 = round(f1, 4)

        # Log results
        print(f"Test Accuracy: {accuracy}")
        print(f"Test F1 Score: {f1}")

        if accuracy >= 0.99:
            print("⚠️ Warning: Accuracy is unusually high. Double-check for overfitting or data leakage.")

        spark.stop()

    except Exception as ex:
        print(f"Error during prediction: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    test_csv = "ValidationDataset.csv"
    model_dir = "trainingmodel.model"
    run_prediction(test_csv, model_dir)
