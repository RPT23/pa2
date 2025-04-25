# Use an official Amazon Linux image as a base
FROM amazonlinux:latest

# Install necessary dependencies
RUN yum update -y && \
    yum install -y python3 java-11-amazon-corretto-devel wget && \
    alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python3 -m ensurepip --upgrade && \
    pip3 install --upgrade pip

# Set environment variables for Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-amazon-corretto.x86_64
ENV PATH=$JAVA_HOME/bin:$PATH

# Create a directory for the application
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create a directory for JAR files
RUN mkdir /libs

# Download required JAR files for Hadoop and AWS SDK
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.1/hadoop-aws-3.3.1.jar -P /libs/ && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1000/aws-java-sdk-bundle-1.11.1000.jar -P /libs/

# Default command to run Spark job
CMD ["spark-submit", "--jars", "/libs/hadoop-aws-3.3.1.jar,/libs/aws-java-sdk-bundle-1.11.1000.jar", "prediction.py"]

