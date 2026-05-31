#######################################################
## first in terminal: pip install transformers torch ##
#######################################################


import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Point dynamically to the unzipped local model folder path
# './final_intent_model' looks for the folder sitting next to this predict.py file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_intent_model")

print("🔄 Loading fine-tuned intent model from local disk...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval() # Put the neural network into evaluation mode
    print("✅ Local model successfully initialized!")
except Exception as e:
    print(f"❌ Error loading model. Make sure the folder path is correct.\nDetails: {e}")
    exit(1)
"""
def get_structured_inference(text, confidence_threshold=0.75):
    # 2. Convert text input into tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # 3. Predict without tracking gradients (keeps it fast on your CPU)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # 4. Turn raw output scores into percentage probabilities
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
    
    intent_scores = []
    for idx, prob in enumerate(probabilities):
        # Gracefully handle integer or string keys from config mapping
        label_name = model.config.id2label.get(idx) or model.config.id2label.get(str(idx))
        intent_scores.append({"intent": label_name, "confidence": round(prob, 2)})
        
    # 5. Sort predictions from highest confidence to lowest
    intent_scores = sorted(intent_scores, key=lambda x: x["confidence"], reverse=True)
    
    top_intent = intent_scores[0]["intent"]
    top_confidence = intent_scores[0]["confidence"]
    top_3_intents = intent_scores[:3]
    
    # Check if confidence threshold is met
    requires_human_review = True if top_confidence < confidence_threshold else False
    
    response_payload = {
        "predicted_intent": top_intent,
        "confidence": top_confidence,
        "top_3_intents": top_3_intents,
        "requires_human_review": requires_human_review
    }
    
    return json.dumps(response_payload, indent=4)
"""

ef get_structured_inference(text, confidence_threshold=0.75):
    # 1. Convert text input into tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # --- THE FIX HAPPENS HERE ---
    # Force remove token_type_ids if they exist, since DistilBERT will crash if it sees them
    inputs.pop("token_type_ids", None)
    # ----------------------------
    
    # 2. Predict without tracking gradients (keeps it fast on your CPU)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # 3. Turn raw output scores into percentage probabilities
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
    
    intent_scores = []
    for idx, prob in enumerate(probabilities):
        # Gracefully handle integer or string keys from config mapping
        label_name = model.config.id2label.get(idx) or model.config.id2label.get(str(idx))
        intent_scores.append({"intent": label_name, "confidence": round(prob, 2)})
        
    # 4. Sort predictions from highest confidence to lowest
    intent_scores = sorted(intent_scores, key=lambda x: x["confidence"], reverse=True)
    
    top_intent = intent_scores[0]["intent"]
    top_confidence = intent_scores[0]["confidence"]
    top_3_intents = intent_scores[:3]
    
    # Check if confidence threshold is met
    requires_human_review = True if top_confidence < confidence_threshold else False
    
    response_payload = {
        "predicted_intent": top_intent,
        "confidence": top_confidence,
        "top_3_intents": top_3_intents,
        "requires_human_review": requires_human_review
    }
    
    return json.dumps(response_payload, indent=4)

# --- 6. Local Command Line Interface Loop ---
if __name__ == "__main__":
    print("\n=======================================================")
    print("      LOCAL INTENT CLASSIFICATION ENGINE READY        ")
    print("      Type 'quit' or 'exit' to stop the script        ")
    print("=======================================================\n")
    
    while True:
        user_input = input("Enter a test customer phrase: ")
        if user_input.lower().strip() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Run prediction and print the pretty JSON block
        json_output = get_structured_inference(user_input)
        print("\nModel Output Response JSON:")
        print(json_output)
        print("-" * 55 + "\n")
