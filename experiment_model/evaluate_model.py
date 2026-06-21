import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

MODEL_PATH = 'experiment_model/naive_bayes_intent.joblib'
TEST_DATA_PATH = 'intent_classification_datasets/test.csv'

print("Loading model and test dataset...")
try:
    model = joblib.load(MODEL_PATH) 
    df_test = pd.read_csv(TEST_DATA_PATH)
    print(f"Model loaded. Found {len(df_test)} test examples.")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Processing Arrays
predictions = []
true_labels = []
latencies = []

print("\nRunning inference benchmarks...")

# Benchmarking Loop
for idx, row in df_test.iterrows():
    text = str(row['text'])
    true_label = str(row['label']).strip()

    start_time = time.perf_counter()
    pred_label = model.predict([text])[0] 
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)

    predictions.append(pred_label)
    true_labels.append(true_label)

# Latency Calculations
avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)
min_latency = np.min(latencies)
max_latency = np.max(latencies)

# Print Report
print("\n==========================================================")
print(" INFERENCE LATENCY BENCHMARK REPORT ")
print("==========================================================")
print(f"Total Queries Tested : {len(df_test)}")
print(f"Minimum Latency : {min_latency:.2f} ms")
print(f"Average Latency (Mean): {avg_latency:.2f} ms")
print(f"95th Percentile (P95) : {p95_latency:.2f} ms")
print(f"99th Percentile (P99) : {p99_latency:.2f} ms")
print(f"Maximum Latency : {max_latency:.2f} ms")
print(f"Throughput Capability : {int(1000 / avg_latency)} requests/second")
print("==========================================================\n")

print("==========================================================")
print(" CLASSIFICATION PERFORMANCE MATRIX ")
print("==========================================================\n")
print(classification_report(true_labels, predictions, digits=4))
print("==========================================================")
