from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
import time
import sys

def prepare_data(input_data):
    try:
        # Update column names to remove quotes if present
        new_columns = [col.replace('"', '') for col in input_data.columns]
        input_data = input_data.toDF(*new_columns)

        label_column = 'quality'

        # Index the 'quality' column
        indexer = StringIndexer(inputCol=label_column, outputCol="label")
        input_data = indexer.fit(input_data).transform(input_data)

        # Select relevant feature columns
        feature_columns = [col for col in input_data.columns if col != label_column]

        # Create a VectorAssembler to combine feature columns
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        # Apply the assembler
        assembled_data = assembler.transform(input_data)

        return assembled_data
    except Exception as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)

def train_model(train_data_path, validation_data_path, output_model):
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("WineQualityTraining") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .getOrCreate()

        bucketname = "rtpmodelbucket"

        train_data = f"s3a://{bucketname}/{train_data_path}"
        validation_data = f"s3a://{bucketname}/{validation_data_path}"

        # Load datasets
        training_raw_data = spark.read.csv(train_data, header=True, inferSchema=True, sep=";")
        validation_raw_data = spark.read.csv(validation_data, header=True, inferSchema=True, sep=";")

        train_data = prepare_data(training_raw_data)
        validation_data = prepare_data(validation_raw_data)

        # Define models
        rf = RandomForestClassifier(labelCol="label", featuresCol="features")
        lr = LogisticRegression(labelCol="label", featuresCol="features")
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
        models = [rf, lr, dt]

        # Create parameter grids
        paramGrids = [
            ParamGridBuilder()
                .addGrid(rf.numTrees, [10, 20, 30])
                .build(),
            ParamGridBuilder()
                .addGrid(lr.maxIter, [10, 20, 30])
                .build(),
            ParamGridBuilder()
                .addGrid(dt.maxDepth, [5, 10, 15])
                .build()
        ]

        # Define evaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )

        results = []
        start_time = time.time()

        for i, model in enumerate(models):
            print(f"Training model {i + 1} ({model.__class__.__name__})")

            # Create a pipeline
            pipeline = Pipeline(stages=[model])

            crossvalidate = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=paramGrids[i],
                evaluator=evaluator,
                numFolds=3
            )

            # Fit the model
            cvModel = crossvalidate.fit(train_data)

            # Predict on validation data
            predictions = cvModel.transform(validation_data)

            # Evaluate metrics
            accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
            recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
            f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

            results.append({
                "Model": model.__class__.__name__,
                "Accuracy": accuracy,
                "Recall": recall,
                "F1 Score": f1_score
            })

        # Save the best model if this is the last iteration
        best_model = cvModel.bestModel

        # Save the best model to S3
        best_model_path = f"s3a://{bucketname}/{output_model}"
        best_model.save(best_model_path)

        # Log overall results
        print("Training completed in {:.2f} seconds".format(time.time() - start_time))

        for result in results:
            print(result)

        # Stop Spark session
        spark.stop()
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_data_path = "TrainingDataset.csv"
    validation_data_path = "ValidationDataset.csv"
    output_model = "trainingmodel.model"

    train_model(train_data_path, validation_data_path, output_model)
