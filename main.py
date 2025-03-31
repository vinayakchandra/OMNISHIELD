import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sklearn.ensemble import IsolationForest
import json

# Initialize FastAPI app
app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["api_logs"]
api_logs_collection = db["api_logs"]

try:
    client.admin.command('ping')
    print("MongoDB connected successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Pydantic Model for API Logs
class APILog(BaseModel):
    endpoint: str
    status_code: int
    response_time: float
    timestamp: str
    client_ip: str

# Validate and store API log
def validate_and_store_log(api_log):
    if 100 <= api_log.status_code <= 599 and api_log.response_time >= 0:
        api_logs_collection.insert_one(api_log.dict())
        return True
    return False

# Preprocess data for model training
def preprocess_data(logs):
    df = pd.DataFrame(logs)
    return df[['response_time']]

# Train and predict anomalies using Isolation Forest
def train_and_predict_model(logs):
    if len(logs) < 10:
        return []
    data = preprocess_data(logs)
    model = IsolationForest(contamination=0.05)
    model.fit(data)
    predictions = model.predict(data)
    anomalies = [logs[i] for i in range(len(logs)) if predictions[i] == -1]
    return anomalies

# API Endpoints
@app.post("/api/logs/")
async def log_api_call(api_log: APILog):
    if not validate_and_store_log(api_log):
        raise HTTPException(status_code=400, detail="Invalid API data")
    return {"message": "API log stored successfully"}

@app.get("/api/check_anomalies/")
async def check_anomalies():
    logs = list(api_logs_collection.find({}, {"_id": 0}))
    anomalies = train_and_predict_model(logs)
    return {"anomalies": anomalies}

@app.post("/api/update_model/")
async def update_model():
    logs = list(api_logs_collection.find({}, {"_id": 0}))
    if len(logs) < 10:
        return {"message": "Not enough data to update model"}
    train_and_predict_model(logs)
    return {"message": "Model updated with new data"}

# Function to visualize the API log data
def visualize_data(logs):
    df = pd.DataFrame(logs)

    # 1. Time Series Plot of Response Times
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['response_time'], label='Response Time', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Response Time (s)')
    plt.title('API Response Times Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # 2. Histogram of Response Times
    plt.figure(figsize=(8, 6))
    plt.hist(df['response_time'], bins=50, alpha=0.7, color='blue', label='Response Times')
    plt.xlabel('Response Time (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of API Response Times')
    plt.legend()
    plt.show()

    # 3. Boxplot for Response Times
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['response_time'], color='blue')
    plt.xlabel('Response Time (s)')
    plt.title('Boxplot of API Response Times')
    plt.show()

    # 4. Scatter Plot of Response Time vs Status Code
    plt.figure(figsize=(8, 6))
    plt.scatter(df['status_code'], df['response_time'], color='blue', alpha=0.5)
    plt.xlabel('Status Code')
    plt.ylabel('Response Time (s)')
    plt.title('Response Time vs Status Code')
    plt.show()

    # 5. Anomaly Detection Summary (Bar Chart)
    anomalies = df[df['is_anomaly'] == 1]
    anomalies_per_endpoint = anomalies['endpoint'].value_counts()

    plt.figure(figsize=(10, 6))
    anomalies_per_endpoint.plot(kind='bar', color='red')
    plt.xlabel('API Endpoint')
    plt.ylabel('Number of Anomalies')
    plt.title('Anomalies Detected per Endpoint')
    plt.xticks(rotation=45)
    plt.show()

    # 6. Correlation Heatmap (if there are multiple features)
    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of API Log Features')
    plt.show()

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

