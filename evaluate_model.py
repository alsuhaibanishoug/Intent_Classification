import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Suppress annoying runtime library initialization warnings
warnings.filterwarnings("ignore")

# Force override transformers initialization block to clear your 2.2.2 environment gatekeeper
import transformers
transformers.utils.import_utils._torch_available = True

# Target file paths - Ensure these match your local directory names
MODEL_DIR = "intent_model_02/model_intent_v2.0.0"
TEST_DATA_PATH = "intent_model_02/datasplit/test.csv"


def load_local_engine():
    """Safely initializes the model and tokenizer directly using underlying PyTorch methods."""
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR]: Target model directory '{MODEL_DIR}' not found.", file=sys.stderr)
        sys.exit(1)
    print("🔄 Loading fine-tuned model and test dataset...")   
    try:
        config = AutoConfig.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        # Instantiate explicitly via structural backend loaders
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
        model.eval()  # Put the neural network into evaluation mode (deactivates dropout)
        
        id2label = config.id2label if config.id2label else {i: f"LABEL_{i}" for i in range(9)}
        label2id = {v: k for k, v in id2label.items()}
       
        return model, tokenizer, id2label, label2id
    except Exception as e:
        print(f"Critical initialization break: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    print("=========================================================")
    print(" Initializing Integrated Performance & Quality Suite...")
    print("=========================================================")
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"[ERROR]: Holdout test dataset file missing at '{TEST_DATA_PATH}'", file=sys.stderr)
        sys.exit(1)
        
    df_test = pd.read_csv(TEST_DATA_PATH)
    model, tokenizer, id2label, label2id = load_local_engine()
    
    text_col = "text" if "text" in df_test.columns else df_test.columns[0]
    label_col = "label" if "label" in df_test.columns else df_test.columns[1]
    
    # Map raw string text fields into ground-truth index integers
    y_true = [label2id[str(lbl)] for lbl in df_test[label_col]]
    print(f"✅ Loaded model from local cache mapping {len(id2label)} intent classes.")
    print(f"⏱️ Processing evaluation benchmarks across {len(df_test)} test rows...\n")
    
    latencies = []
    y_pred = []
    y_prob_list = []  # New container tracking full 2D probability matrices for AUC
    
    # --- Execute Evaluation & Core Timing Loops ---
    for _, row in df_test.iterrows():
        phrase = str(row[text_col])
        
        # Tokenize incoming text queries
        inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=128)
        inputs.pop("token_type_ids", None) # Safe strip function to clear DistilBERT format errors
        
        # Microsecond precision delta tracking point
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            # Pass raw activation output scores through Softmax to get precise percentages
            probabilities = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
            
        end_time = time.perf_counter()
        
        # Record tracking metrics
        latencies.append((end_time - start_time) * 1000)
        y_pred.append(int(np.argmax(probabilities)))
        y_prob_list.append(probabilities)
        
    # Convert probability vector listings into a strict 2D matrix shape [Num_Samples, 9]
    y_prob_matrix = np.array(y_prob_list)
    
    # --- Compute Metric Averages ---
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)  # 95% of requests are faster than this
    p99_latency = np.percentile(latencies, 99)  # 99% of requests are faster than this
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    throughput = 1000 / avg_latency
    
    # --- Calculate Multi-Class ROC-AUC via One-Vs-Rest (OvR) ---
    macro_auc = roc_auc_score(y_true, y_prob_matrix, multi_class='ovr', average='macro')
    weighted_auc = roc_auc_score(y_true, y_prob_matrix, multi_class='ovr', average='weighted')
    
    # --- Print Unified Production Dashboard Output Logs ---
    print("=" * 65)
    print(" 🚀 INFERENCE LATENCY BENCHMARK REPORT")
    print("=" * 65)
    print(f"Minimum Latency       : {min_latency:.2f} ms")
    print(f"Average Latency (Mean): {avg_latency:.2f} ms")
    print(f"95th Percentile (P95) : {p95_latency:.2f} ms")
    print(f"99th Percentile (P99) : {p99_latency:.2f} ms")
    print(f"Maximum Latency       : {max_latency:.2f} ms")
    print(f"Host Core Throughput  : {throughput:.1f} live client queries/second per core")
    print("=" * 65)
    
    print("\n" + "=" * 65)
    print(" 📈 INTEGRATED CLASSIFICATION PERFORMANCE MATRIX")
    print("=" * 65)
    print(f" Final System Test Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f" Macro-Averaged ROC-AUC     : {macro_auc:.4f}  (Evaluates boundaries equally)")
    print(f" Weighted-Averaged ROC-AUC  : {weighted_auc:.4f}  (Adjusts for class frequencies)")
    print("=" * 65)
    
    print("\n Detailed Class-by-Class Performance Matrix Breakdown:")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


if __name__ == "__main__":
    main()