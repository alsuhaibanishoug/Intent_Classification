import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "intent_model"
MAX_LENGTH = 64
CONFIDENCE_THRESHOLD = 0.75

print("🔄 Loading fine-tuned intent model from local disk...")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    print(f"✅ Local model successfully initialized on {device}!")
except Exception as e:
    print(f"❌ Error loading model. Make sure the folder path is correct.\nDetails: {e}")
    exit(1)


def tokenize_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def warmup_model():
    warmup_texts = [
        "I want a refund",
        "Where is my order?",
        "I cannot log in",
        "My payment failed",
        "Please give me the national crisis line number"
    ]

    with torch.inference_mode():
        for text in warmup_texts:
            inputs = tokenize_text(text)
            _ = model(**inputs)


def get_structured_inference(text, confidence_threshold=CONFIDENCE_THRESHOLD):
    inputs = tokenize_text(text)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits

    if device.type == "cuda":
        torch.cuda.synchronize()

    probabilities = F.softmax(logits, dim=-1).squeeze().detach().cpu().tolist()

    intent_scores = []
    for idx, prob in enumerate(probabilities):
        label_name = model.config.id2label.get(idx) or model.config.id2label.get(str(idx))
        intent_scores.append({
            "intent": label_name,
            "confidence": round(float(prob), 4)
        })

    intent_scores = sorted(intent_scores, key=lambda x: x["confidence"], reverse=True)

    top_intent = intent_scores[0]["intent"]
    top_confidence = intent_scores[0]["confidence"]
    requires_human_review = top_confidence < confidence_threshold

    response_payload = {
        "predicted_intent": top_intent,
        "confidence": top_confidence,
        "top_3_intents": intent_scores[:3],
        "requires_human_review": requires_human_review
    }

    return json.dumps(response_payload, indent=4)


if __name__ == "__main__":
    print("🔥 Warming up model...")
    warmup_model()

    print("\n=======================================================")
    print("      LOCAL INTENT CLASSIFICATION ENGINE READY        ")
    print("      Type 'quit' or 'exit' to stop the script        ")
    print("=======================================================\n")

    while True:
        user_input = input("Enter a test customer phrase: ")

        if user_input.lower().strip() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        json_output = get_structured_inference(user_input)
        print("\nModel Output Response JSON:")
        print(json_output)
        print("-" * 55 + "\n")