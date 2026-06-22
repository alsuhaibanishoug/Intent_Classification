# Fahim - فهيم
## Multi-Class Intent Classification Engine (DistilBERT)

A high-performance, low-latency, and cost-efficient intent classification system built on top of **DistilBERT** (`distilbert-base-uncased`). This engine dynamically maps user storefront queries into 9 distinct business intent classes. 

Our core architecture is optimized for lightweight CPU deployment on local machines, achieving sub-6 millisecond inference speeds with 92% test classification accuracy.

---

## 📈 System Metrics Summary

*   **Average Latency:** `4.74 ms` on standard MacBook CPU
*   **P95 Operational Latency:** `5.88 ms` (95% of customer queries process under 6ms)
*   **Throughput Capability:** ~210 requests per second per core
*   **Final Test Accuracy:** `0.9213` across all 9 target intents
*   **ROC-AUC:** `0.9901` across all 9 target intents

---

## 📂 Repository Structure

```text
├── notebooks/                                    # Research and development notebooks
│   └── fineTuning_distillBERT.ipynb              # Dataset cleaning, 3-way split code, training data augmentation script, and Fine-Tuning pipeline setup
├── predict.py                                    # Live interactive local inference CLI script
├── evaluate_model.py                             # Latency benchmarking & performance reporting script
├── ui_app.py                                     # Streamlit web interface script
└── README.md                                     # This file
```

---

## 🚀 Local Deployment: How to Run Inference

If you would like to clone this repository and run live intent classification predictions locally on your computer, follow these quick setup instructions:

### Step 1: Install Dependencies
Open your local Terminal or Command Prompt and install the lightweight CPU versions of the required machine learning packages:

```bash
pip install -r requirements.txt
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
## ✨ Streamlit Web Application

An interactive web application built with Streamlit that allows users to classify customer service messages in real time using the fine-tuned DistilBERT model.

### 🌟 Features
- 🎨 **Modern User Interface**
  - Gradient backgrounds and animated components.
  - Clean and responsive layout.

- ✅ **Live Input Validation**
  - Validates user input before classification.
  - Displays clear error messages and requirement checks.

- 🤖 **Real-Time Intent Classification**
  - Predicts one of the nine customer service intents.
  - Displays the primary predicted intent with confidence score.

- 📊 **Prediction Analytics**
  - Top 3 predicted intents with confidence percentages.
  - Intent probability distribution charts.
  - Inference time and prediction timestamp.

- 🚩 **Human Review Flag**
  - Automatically flags low-confidence predictions for manual review.

- 🎯 **Priority Detection**
  - Displays message urgency levels (Low, Medium, High, Critical).

- 📱 **Responsive Design**
  - Works on desktop and mobile devices.

---

## 🚀 Running the Streamlit Application

### 1. Install Dependencies:
```bash
pip install "streamlit>=1.28.0"
```

### 2. Launch the Application
```bash
streamlit run ui_app.py
```

The application will open automatically at:

```text
http://localhost:8501
```

---

## 🖥️ How to Use

### Step 1 – Enter a Message
Type or paste a customer support message into the text box.

### Step 2 – Validate Input
The application performs live validation and ensures that all requirements are satisfied.

### Step 3 – Classify
Click the **Classify** button to generate a prediction.

### Step 4 – Review Results
The application displays:

- Predicted intent
- Confidence score
- Priority level
- Top 3 candidate intents
- Human review recommendation (if needed)
- Inference latency and timestamp

### Step 5 – Try Another Message
Click **Clear** to reset the form and classify another message.

---

## 📝 License
Created by **Fahim Team** © 2026

---
## 📖 Deep-Dive Project Wiki Pages

For a thorough breakdown of our entire engineering workflow, datasets, and regularization strategies, review our dedicated project wiki entries:

1. **[Wiki Part 1: Data Preparation Stage](https://github.com/alsuhaibanishoug/Intent_Classification/wiki/Wiki-Part-1:-Data-Preparation-Stage)**
    * Covers the initial Chinese-to-English translation loop.
    * Documents structural category relabeling matching the client brief.
    * Explains our zero-shot missing category synthesis and Masked Language Modeling (MLM) data augmentation rules to reach our balanced dataset layout.
2. **[Wiki Part 2: Model Architecture & Fine-Tuning Pipeline Guide](https://github.com/alsuhaibanishoug/Intent_Classification/wiki/Wiki-Part-2:-Model-Architecture-&-Fine%E2%80%90Tuning-Pipeline-Guide)**
    * Outlines our overfitting defense safeguards: Weight Decay (`0.01`) and `EarlyStoppingCallback` circuit breakers.
    * Provides the complete post-execution training performance logs.
3. **[Wiki Part 3: Local Evaluation & Latency Benchmarks Guide](https://github.com/alsuhaibanishoug/Intent_Classification/wiki/Wiki-Part-3:-Local-Evaluation-&-Latency-Benchmarks-Guide)**
    * Documents our local CPU cache warm-up and time-delta evaluation strategy.
    * Contains the final complete classification matrix table mapping 99% ROC-AUC metrics across our test environment.
