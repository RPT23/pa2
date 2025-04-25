import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPrediction {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("WineQualityPrediction");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("WineQualityPrediction").getOrCreate();

        // Load the trained model
        PipelineModel model = PipelineModel.load("s3://<your-s3-bucket>/wineQualityModel");

        // Load the test dataset from S3
        Dataset<Row> testData = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("s3://<your-s3-bucket>/TestDataset.csv");

        // Make predictions on the test dataset
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate the model performance on the test dataset
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 score on test data: " + f1Score);

        jsc.close();
    }
}