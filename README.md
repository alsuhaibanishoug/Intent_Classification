# Fahim - فهيم
## Multi-Class Intent Classification Engine (DistilBERT)

A high-performance, low-latency, and cost-efficient intent classification system built on top of **DistilBERT** (`distilbert-base-uncased`). This engine dynamically maps user storefront queries into 9 distinct business intent classes. 

Our core architecture is optimized for lightweight CPU deployment on local machines, achieving sub-5 millisecond inference speeds with 92% test classification accuracy.

---

## 📈 System Metrics Summary

*   **Average Latency:** `3.96 ms` on standard MacBook CPU
*   **P95 Operational Latency:** `4.77 ms` (95% of customer queries process under 5ms)
*   **Throughput Capability:** ~252 requests per second per core
*   **Final Test Accuracy:** `0.9213` across all 9 target intents
*   **ROC-AUC:** `0.9901` across all 9 target intents

---

## 📂 Repository Structure

```text
├── experiment_model/                             # Saved model assets or experiment outputs
├── notebooks/                                    # Research and development notebooks
│   └── fineTuning_distillBERT.ipynb              # Dataset cleaning, 3-way split code, training data augmentation script, and Fine-Tuning pipeline setup
├── intent_classification_datasets/               # Full data pipeline evolution directory
│   ├── original_dataset.csv                      # Translated client data after Structural Audit & Relabeling
│   ├── synthetic_expanded_dataset.csv            # Generated text samples after Zero-Shot Missing Category Generation
│   ├── data_new.csv                              # Cleaned dataset after Distribution Audit & Second-Pass Balancing
│   ├── train_auhmented.csv                       # Augmented training data split 
│   ├── val.csv                                   # Validation split on data_new.csv
│   └── test.csv                                  # Test split on data_new.csv
├── predict.py                                    # Live interactive local inference CLI script
├── evaluate_model.py                             # Latency benchmarking & performance reporting script
└── README.md                                     # This file
```

---

## 🚀 Local Deployment: How to Run Inference

If you would like to clone this repository and run live intent classification predictions locally on your computer, follow these quick setup instructions:

### Step 1: Install Dependencies
Open your local Terminal or Command Prompt and install the lightweight CPU versions of the required machine learning packages:

```bash
pip install transformers torch pandas scikit-learn
```
*(Note: Using standard DistilBERT architectures with older local environment parameters may require a `transformers==4.40.0` environment match).*

### Step 2: Download & Extract the Model Files
1. Ensure the `intent_model`([which lives here](https://github.com/alsuhaibanishoug/Intent_Classification/releases/download/v2.0.1/intent_model.zip)) directory is unzipped and placed in the root folder of this project.
2. Confirm the directory contains `config.json`, `model.safetensors`, and `tokenizer_config.json` inside it.

### Step 3: Run the Live Interactive CLI
To launch the interactive command-line tool where you can test your custom storefront phrases:

```bash
python predict.py
```
**Example Usage Input & Output Payload:**
```text
Enter a test customer phrase: When will my shipment arrive?

Model Output Response JSON:
{
    "predicted_intent": "logistics",
    "confidence": 0.99,
    "top_3_intents": [
        {"intent": "logistics", "confidence": 0.99},
        {"intent": "product_inquiry", "confidence": 0.01},
        {"intent": "general_question", "confidence": 0.00}
    ],
    "requires_human_review": false
}
```

### Step 4: Run the Speed & Accuracy Evaluation Suite
To re-verify our system latency benchmarks and confirm the classification matrix report over our 1,191 row holdout test data asset:

```bash
python evaluate_model.py
```

---

## 📖 Deep-Dive Project Wiki Pages

For a thorough breakdown of our entire engineering workflow, datasets, and regularization strategies, review our dedicated project wiki entries:

1. **[Wiki Part 1: Data Preparation Stage](https://github.com/alsuhaibanishoug/Intent_Classification/wiki/Wiki-Part-1:-Data-Preparation-Stage)**
    * Covers the initial Chinese-to-English translation loop.
    * Documents structural category relabeling matching the client brief.
    * Explains our zero-shot missing category synthesis and Masked Language Modeling (MLM) data augmentation rules to reach our balanced dataset layout.
2. **[Wiki Part 2: Model Architecture & Fine-Tuning Pipeline Guide](https://github.com/alsuhaibanishoug/Intent_Classification/wiki/Wiki-Part-2:-Model-Architecture-&-Fine%E2%80%90Tuning-Pipeline-Guide)**
    * Detailing why we chose Transfer Learning over Classical ML or training Deep Learning layers from scratch.
    * Outlines our overfitting defense safeguards: Weight Decay (`0.01`) and `EarlyStoppingCallback` circuit breakers.
    * Provides the complete post-execution training performance logs.
3. **[Wiki Part 3: Local Evaluation & Latency Benchmarks Guide](https://github.com/alsuhaibanishoug/Intent_Classification/wiki/Wiki-Part-3:-Local-Evaluation-&-Latency-Benchmarks-Guide)**
    * Documents our local CPU cache warm-up and time-delta evaluation strategy.
    * Contains the final complete classification matrix table mapping 99% ROC-AUC metrics across our test environment.
