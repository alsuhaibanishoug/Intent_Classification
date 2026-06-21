import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score
)

warnings.filterwarnings("ignore")

MODEL_DIR = "intent_model"
TEST_DATA_PATH = "intent_classification_datasets/test.csv"

MAX_LENGTH = 64
BATCH_SIZE = 16
WARMUP_RUNS = 10


def load_local_engine():
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR]: Target model directory '{MODEL_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    print("🔄 Loading fine-tuned model and test dataset...")

    config = AutoConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = config.id2label if config.id2label else {i: f"LABEL_{i}" for i in range(config.num_labels)}
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    return model, tokenizer, id2label, label2id, device


def tokenize_batch(tokenizer, texts, device):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def warmup_model(model, tokenizer, device):
    warmup_texts = [
        "I want a refund",
        "Where is my order?",
        "I cannot log in",
        "My payment failed",
        "What are your business hours?"
    ]

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            inputs = tokenize_batch(tokenizer, warmup_texts, device)
            _ = model(**inputs)


def main():
    print("=========================================================")
    print(" Initializing Integrated Performance & Quality Suite...")
    print("=========================================================")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"[ERROR]: Holdout test dataset file missing at '{TEST_DATA_PATH}'", file=sys.stderr)
        sys.exit(1)

    df_test = pd.read_csv(TEST_DATA_PATH)

    model, tokenizer, id2label, label2id, device = load_local_engine()

    text_col = "text" if "text" in df_test.columns else df_test.columns[0]
    label_col = "label" if "label" in df_test.columns else df_test.columns[1]

    missing_labels = sorted(set(df_test[label_col].astype(str)) - set(label2id.keys()))
    if missing_labels:
        print(f"[ERROR]: Labels in test file not found in model config: {missing_labels}", file=sys.stderr)
        sys.exit(1)

    y_true = [label2id[str(lbl)] for lbl in df_test[label_col]]

    print(f"✅ Loaded model from local cache mapping {len(id2label)} intent classes.")
    print(f"🖥️ Running on device: {device}")
    print(f"⏱️ Processing evaluation benchmarks across {len(df_test)} test rows...\n")

    print("🔥 Running warm-up predictions...")
    warmup_model(model, tokenizer, device)

    latencies = []
    y_pred = []
    y_prob_list = []

    texts = df_test[text_col].astype(str).tolist()

    for start_idx in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start_idx:start_idx + BATCH_SIZE]

        inputs = tokenize_batch(tokenizer, batch_texts, device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        batch_latency_ms = (end_time - start_time) * 1000
        per_sample_latency_ms = batch_latency_ms / len(batch_texts)

        latencies.extend([per_sample_latency_ms] * len(batch_texts))
        y_pred.extend(np.argmax(probabilities, axis=1).astype(int).tolist())
        y_prob_list.extend(probabilities)

    y_prob_matrix = np.array(y_prob_list)

    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    throughput = 1000 / avg_latency
    
    overall_precision = precision_score(y_true,y_pred,average="macro",zero_division=0)
    overall_recall = recall_score(y_true,y_pred,average="macro",zero_division=0)
    macro_auc = roc_auc_score(y_true, y_prob_matrix, multi_class="ovr", average="macro")
    weighted_auc = roc_auc_score(y_true, y_prob_matrix, multi_class="ovr", average="weighted")

    print("=" * 65)
    print(" 🚀 INFERENCE LATENCY BENCHMARK REPORT")
    print("=" * 65)
    print(f"Total Queries Tested  : {len(df_test)}")
    print(f"Minimum Latency        : {min_latency:.2f} ms/sample")
    print(f"Median Latency         : {median_latency:.2f} ms/sample")
    print(f"Average Latency Mean   : {avg_latency:.2f} ms/sample")
    print(f"95th Percentile P95    : {p95_latency:.2f} ms/sample")
    print(f"99th Percentile P99    : {p99_latency:.2f} ms/sample")
    print(f"Maximum Latency        : {max_latency:.2f} ms/sample")
    print(f"Estimated Throughput   : {throughput:.1f} samples/second")
    print("=" * 65)

    print("\n" + "=" * 65)
    print(" 📈 INTEGRATED CLASSIFICATION PERFORMANCE MATRIX")
    print("=" * 65)
    print(f" Final System Test Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f" Overall Macro Precision    : {overall_precision:.4f}")
    print(f" Overall Macro Recall       : {overall_recall:.4f}")
    print(f" Macro-Averaged ROC-AUC     : {macro_auc:.4f}")
    print(f" Weighted-Averaged ROC-AUC  : {weighted_auc:.4f}")
    print("=" * 65)
    
    target_names = [id2label[i] for i in range(len(id2label))]

    print("\n Detailed Class-by-Class Performance Matrix Breakdown:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    results_df = df_test.copy()
    results_df["true_label"] = [id2label[i] for i in y_true]
    results_df["predicted_label"] = [id2label[i] for i in y_pred]
    results_df["correct"] = results_df["true_label"] == results_df["predicted_label"]

    print("\n Misclassified Examples Across All Labels:")

    all_errors = results_df[results_df["correct"] == False].copy()

    if len(all_errors) == 0:
        print("No misclassifications found.")
    else:
        print(
            all_errors[
                [text_col, "true_label", "predicted_label"]
            ].to_string(index=False)
        )

    print("\n Misclassification Pairs:")

    if len(all_errors) == 0:
        print("No error pairs found.")
    else:
        error_pairs = (
            all_errors
            .groupby(["true_label", "predicted_label"])
            .size()
            .reset_index(name="count")
            .sort_values(["true_label", "count"], ascending=[True, False])
        )

        print(error_pairs.to_string(index=False))

if __name__ == "__main__":
    main()