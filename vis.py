import matplotlib.pyplot as plt
import pandas as pd

# Assuming logs is a DataFrame with timestamp and response_time columns
df = pd.DataFrame(logs)

# Plotting response times over time
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['response_time'], label='Response Time', color='blue')

# Highlight anomalies
anomalies = df[df['is_anomaly'] == 1]
plt.scatter(anomalies['timestamp'], anomalies['response_time'], color='red', label='Anomalies')

plt.xlabel('Time')
plt.ylabel('Response Time (s)')
plt.title('API Response Times with Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.show()
