
FROM openjdk:8
WORKDIR /app
COPY . /app
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://downloads.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz && \
    tar -xvzf spark-3.5.0-bin-hadoop3.tgz && \
    mv spark-3.5.0-bin-hadoop3 spark
CMD ["java", "-cp", ".:spark/jars/*", "WineModelValidator"]
