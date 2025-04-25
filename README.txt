
# Wine Predictor (CS643 - PA2)

## 1. How to Train the Model (Spark)
```bash
spark-submit --class WineModelTrainer --master local[*] WineModelTrainer.java
```

## 2. How to Validate the Model
```bash
spark-submit --class WineModelValidator --master local[*] WineModelValidator.java
```

## 3. How to Build Docker Image
```bash
docker build -t wine-predictor .
```

## 4. How to Run Prediction using Docker
```bash
docker run wine-predictor
```

## 5. Push Docker to Docker Hub
```bash
docker login
docker tag wine-predictor yourusername/wine-predictor:v1
docker push yourusername/wine-predictor:v1
```

## Notes
- Use the training and validation datasets provided.
- The application prints F1 score.
