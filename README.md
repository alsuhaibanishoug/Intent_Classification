# Fahim - فهيم
## Multi-Class Intent Classification Engine (DistilBERT)

A high-performance, low-latency, and cost-efficient intent classification system built on top of **DistilBERT** (`distilbert-base-uncased`). This engine dynamically maps user storefront queries into 9 distinct business intent classes. 

Our core architecture is optimized for lightweight CPU deployment on local machines, achieving sub-16 millisecond inference speeds with 100% test classification accuracy.

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
├── Notebooks/                                    # Research and development notebooks
│   └── fineTuning_distillBERT.ipynb              # Dataset cleaning, 3-way split code, training data augmentation script, and Fine-Tuning pipeline setup
├── intent_model/                                 # Unzipped trained model binaries (config.json, weights)
├── intent_classification_datasets/               # Full data pipeline evolution directory
│   ├── original_dataset.csv                      # Translated client data after Structural Audit & Relabeling
│   ├── synthetic_expanded_dataset.csv            # Generated text samples after Zero-Shot Missing Category Generation
│   ├── combined_dataset.csv                      # The initial merge of the past two datasets combined
│   ├── data_new.csv                              # Cleaned dataset after Distribution Audit & Second-Pass Balancing
│   ├── data_new_augmented_dataset.csv            # Final expanded dataset after Contextual MLM Augmentation
│   └── test_cleaned.csv                          # 89 rows of clean holdout test dataset
├── predict.py                                    # Live interactive local inference CLI script
├── evaluate_model.py                             # Latency benchmarking & performance reporting script
└── README.md                                     # This file
```

---

## 📊 Dataset Pipeline Breakdown

The `intent_classification_datasets/` folder contains the step-by-step evolution of our training data:

1.  **`original_dataset.csv`**: The translated client data after our *Structural Audit & Relabeling* phase, mapping unstructured business rows to concrete engineering intent categories.
2.  **`synthetic_expanded_dataset.csv`**: The baseline text entries generated during our *Zero-Shot Missing Category Generation* pass to create training data for missing target intents.
3.  **`combined_dataset.csv`**: The unified merge combining both the mapped client rows and the first-pass generated missing categories.
4.  **`data_new.csv`**: The dataset after our *Distribution Audit & Second-Pass Balancing* pass, bringing all categories to a perfectly even starting foundation of 25 rows per class.
5.  **`data_new_augmented_dataset.csv`**: Our final production training data created after running the *Contextual Masked Language Modeling (MLM) Augmentation* engine, scaling our balanced rows up to a comprehensive, high-volume corpus.

---

## 📓 Jupyter Notebooks Overview

The developmental phase of this project is separated into three clean notebooks located inside the `notebooks/` directory:

1.  **`data_augmentation.ipynb`**: Hosts our automated Masked Language Modeling (MLM) data augmentation engine. It leverages a pre-trained model to safely scale each target class up toward our 1,000 balanced rows goal while protecting key intent keywords from modification.
2.  **`augmented_dataset_analysis.ipynb`**: Handles dataset diagnostics and data health checks. It performs lowercase string cleaning, strips layout noise, scans for semantic cross-label text conflicts, maps string labels to integer tracking IDs, and generates our stratified 70/15/15 train, validation, and test splits.
3.  **`fineTuning_distillBERT.ipynb`**: Handles the final PyTorch fine-tuning workflow loop. It sets up the gradient optimization pipeline over our clean data splits, injects weight decay regularization to stop text memorization, and manages training termination via early stopping callbacks.

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
1. Ensure the `intent_model` directory is unzipped and placed in the root folder of this project.
2. Confirm the directory contains `config.json`, `model.safetensors`([which lives here](https://github.com/alsuhaibanishoug/Intent_Classification/releases/tag/v1.0.1)), and `tokenizer_config.json` inside it.

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
    * Outlines our overfitting defense safeguards: Weight Decay (`0.1`) and `EarlyStoppingCallback` circuit breakers.
    * Provides the complete post-execution training performance logs.
3. **[Wiki Part 3: Local Evaluation & Latency Benchmarks Guide](https://github.com/alsuhaibanishoug/Intent_Classification/wiki/Wiki-Part-3:-Local-Evaluation-&-Latency-Benchmarks-Guide)**
    * Documents our local CPU cache warm-up and time-delta evaluation strategy.
    * Contains the final complete classification matrix table mapping 100% precision and recall metrics across our test environment.

