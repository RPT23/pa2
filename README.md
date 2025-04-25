
# Wine Quality Prediction - CS643 Programming Assignment 2

This project implements a wine quality prediction system using Apache Spark MLlib on AWS EC2 instances. The model is trained in parallel on a Spark cluster and deployed using Docker.

## ✅ Features
- Parallel model training on 4 EC2 instances using Spark
- Wine quality classification using Logistic Regression
- Model evaluation via F1 score
- Model deployed in a Docker container

## 📁 Dataset
- `TrainingDataset.csv`: Used to train the model
- `ValidationDataset.csv`: Used to evaluate and tune the model
- `TestDataset.csv`: Used by the instructor to grade predictions

## 🚀 How to Run

### 1. Train Model on Spark Cluster
```bash
spark-submit --class WineQualityTraining --master spark://<master-node-ip>:7077 target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar
```

### 2. Predict with Trained Model
```bash
spark-submit --class WineQualityPrediction --master local[*] target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar
```

### 3. Docker Image
Create a Dockerfile like:
```
FROM openjdk:8-jre-slim
COPY target/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar /app/
CMD ["java", "-jar", "/app/wine-quality-1.0-SNAPSHOT-jar-with-dependencies.jar"]
```

Build and push Docker image:
```bash
docker build -t wine-quality-prediction .
docker tag wine-quality-prediction <your-ecr-repo-url>
docker push <your-ecr-repo-url>
```

Run the Docker container:
```bash
docker run wine-quality-prediction
```

## 🧠 Output
Displays F1 score based on predictions from the model loaded from S3.

---
CS643 - Cloud Computing, NJIT
