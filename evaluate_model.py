import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = './intent_model'
TEST_DATA_PATH = 'intent_classification_datasets/test_cleaned.csv'

print("🔄 Loading fine-tuned model and test dataset...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()  # Put network in evaluation mode
    
    df_test = pd.read_csv(TEST_DATA_PATH)
    print(f"✅ Model loaded. Found {len(df_test)} test examples.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")
    exit(1)

# Extract Dynamic Mappings from Saved Model Config
unique_labels = sorted(list(model.config.id2label.values()))
label2id = {label: idx for idx, label in enumerate(unique_labels)}

# Processing Arrays
predictions = []
true_labels = []
latencies = []

print("\n⏱️ Running inference benchmarks across test suite...")

# Warm-up run to initialize CUDA/CPU cache so the first row doesn't skew benchmarks
_ = tokenizer("warmup text", return_tensors="pt")
with torch.no_grad():
    _ = model(**{k: v for k, v in _.items() if k != "token_type_ids"})

# Benchmarking Loop
for idx, row in df_test.iterrows():
    text = str(row['text'])
    true_label = str(row['label']).strip()
    
    # Measure Latency strictly around Tokenization + Model Forward Pass
    start_time = time.perf_counter()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs.pop("token_type_ids", None) 
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
    pred_idx = np.argmax(probabilities)
    
    end_time = time.perf_counter()
    
    # Calculate execution time in milliseconds
    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)
    
    # Map numerical prediction back to string label
    pred_label = model.config.id2label.get(pred_idx) or model.config.id2label.get(str(pred_idx))
    
    predictions.append(pred_label)
    true_labels.append(true_label)

# Latency Calculations
avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)  # 95% of requests are faster than this
p99_latency = np.percentile(latencies, 99)  # 99% of requests are faster than this
min_latency = np.min(latencies)
max_latency = np.max(latencies)

# Performance Statistics Report
print("\n==========================================================")
# Metrics summary
print("             INFERENCE LATENCY BENCHMARK REPORT           ")
print("==========================================================")
print(f"Total Queries Tested  : {len(df_test)}")
print(f"Minimum Latency       : {min_latency:.2f} ms")
print(f"Average Latency (Mean): {avg_latency:.2f} ms")
print(f"95th Percentile (P95) : {p95_latency:.2f} ms")
print(f"99th Percentile (P99) : {p99_latency:.2f} ms")
print(f"Maximum Latency       : {max_latency:.2f} ms")
print(f"Throughput Capability : {int(1000 / avg_latency)} requests/second (approx)")
print("==========================================================\n")

print("==========================================================")
print("             CLASSIFICATION PERFORMANCE MATRIX            ")
print("==========================================================\n")

# Filters out any labels that might not be explicitly represented in your test split subset
present_labels = sorted(list(set(true_labels) | set(predictions)))

print(classification_report(
    true_labels,
    predictions,
    labels=present_labels,
    digits=4
))
print("==========================================================")
