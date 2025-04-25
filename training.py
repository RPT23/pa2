from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time
import sys

def preprocess_data(df):
    try:
        cleaned_cols = [col.replace('"', '') for col in df.columns]
        df = df.toDF(*cleaned_cols)

        label_indexer = StringIndexer(inputCol="quality", outputCol="label")
        df = label_indexer.fit(df).transform(df)

        features = [col for col in df.columns if col not in ("quality", "label")]
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        df = assembler.transform(df)

        return df
    except Exception as err:
        print(f"Data processing error: {err}")
        sys.exit(1)

def train_and_evaluate(train_file, valid_file, model_output):
    try:
        spark = SparkSession.builder \
            .appName("WineQualityGBTModel") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .getOrCreate()

        bucket = "rtpmodelbucket"
        train_path = f"s3a://{bucket}/{train_file}"
        valid_path = f"s3a://{bucket}/{valid_file}"

        train_raw = spark.read.csv(train_path, header=True, inferSchema=True, sep=";")
        valid_raw = spark.read.csv(valid_path, header=True, inferSchema=True, sep=";")

        train_df = preprocess_data(train_raw)
        valid_df = preprocess_data(valid_raw)

        # Only use GBTClassifier
        gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

        param_grid = ParamGridBuilder() \
            .addGrid(gbt.maxIter, [10, 20, 30]) \
            .addGrid(gbt.maxDepth, [3, 5, 7]) \
            .build()

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )

        pipeline = Pipeline(stages=[gbt])

        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3
        )

        start_time = time.time()

        cv_model = crossval.fit(train_df)
        preds = cv_model.transform(valid_df)

        accuracy = evaluator.evaluate(preds, {evaluator.metricName: "accuracy"})
        recall = evaluator.evaluate(preds, {evaluator.metricName: "weightedRecall"})
        f1_score = evaluator.evaluate(preds, {evaluator.metricName: "f1"})

        print("Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

        best_model = cv_model.bestModel
        model_path = f"s3a://{bucket}/{model_output}"
        best_model.save(model_path)

        print(f"Training completed in {time.time() - start_time:.2f} seconds")

        spark.stop()

    except Exception as ex:
        print(f"Training error: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    train_file = "TrainingDataset.csv"
    valid_file = "ValidationDataset.csv"
    model_output = "gbt_trained_model_output"

    train_and_evaluate(train_file, valid_file, model_output)
