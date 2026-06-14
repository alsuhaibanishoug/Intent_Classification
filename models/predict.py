#!/usr/bin/env python3
"""
Predict CLI Script for Intent Classification Engine
Handles standard Pickle (.pkl) and Joblib pipelines seamlessly.
"""

import os
import sys
import json
import warnings

# Suppress unnecessary library warnings during runtime initialization
warnings.filterwarnings("ignore")

# Define the relative path to your pickled model asset file
MODEL_PATH = 'models/logreg_pipeline.pkl'


def load_model_pipeline(path):
    """
    Safely opens and deserializes the trained model pipeline file,
    handling both Joblib and classic Pickle serialization.
    """
    if not os.path.exists(path):
        print(f"Error: Model artifact file not found at path: '{path}'", file=sys.stderr)
        sys.exit(1)
        
    # Check for Git LFS placeholder contamination
    try:
        with open(path, "r", errors="ignore") as text_file:
            head = text_file.read(100)
            if "git-lfs" in head or "version https://git-lfs" in head:
                print("\n[CRITICAL ERROR]: Your model file is a Git LFS pointer text file, not the actual model binary!", file=sys.stderr)
                print("Please download the raw binary file directly from GitHub or run 'git lfs pull'.\n", file=sys.stderr)
                sys.exit(1)
    except Exception:
        pass # Not a text file, safe to proceed with binary loading

    # Try loading with joblib first (Highly recommended for Scikit-Learn pipelines)
    try:
        import joblib
        with open(path, "rb") as file:
            return joblib.load(file)
    except Exception:
        # If joblib fails or isn't installed, fall back to native pickle
        try:
            import pickle
            with open(path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            print(f"\n[CRITICAL FAILURE]: Could not deserialize the model pipeline.", file=sys.stderr)
            print(f"Error details: {e}", file=sys.stderr)
            print("Suggestions:\n 1. Ensure the file wasn't corrupted during unzipping/transfer.\n 2. Verify which library saved the model (scikit-learn, joblib, pickle, pytorch?).\n", file=sys.stderr)
            sys.exit(1)


def generate_json_payload(pipeline, user_text, confidence_threshold=0.60):
    """
    Processes the raw text phrase through the ML model pipeline.
    """
    predicted_intent = str(pipeline.predict([user_text])[0])
    
    try:
        classes = pipeline.classes_
        probabilities = pipeline.predict_proba([user_text])[0]
        
        intent_probs = [
            {"intent": str(intent), "confidence": round(float(prob), 2)}
            for intent, prob in zip(classes, probabilities)
        ]
        intent_probs = sorted(intent_probs, key=lambda x: x["confidence"], reverse=True)
        top_3_intents = intent_probs[:3]
        
        primary_confidence = next(
            item["confidence"] for item in intent_probs if item["intent"] == predicted_intent
        )
    except (AttributeError, IndexError):
        top_3_intents = [{"intent": predicted_intent, "confidence": 1.00}]
        primary_confidence = 1.00

    requires_human_review = primary_confidence < confidence_threshold

    return {
        "predicted_intent": predicted_intent,
        "confidence": primary_confidence,
        "top_3_intents": top_3_intents,
        "requires_human_review": requires_human_review
    }


def main():
    print("=" * 65)
    print(" Initializing Multi-Class Intent Classification Engine...")
    print("=" * 65)
    
    intent_pipeline = load_model_pipeline(MODEL_PATH)
    
    print("\n System ready! Enter a test phrase to run live model classification.")
    print(" Type 'exit' or 'quit' at any time to close the terminal session.\n")
    
    while True:
        try:
            user_input = input("Enter a test customer phrase: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("\nShutting down session. Goodbye!")
                break
                
            if not user_input:
                print(" Warning: Empty sentence string. Please enter a valid customer phrase.\n")
                continue
                
            output_data = generate_json_payload(intent_pipeline, user_input)
            
            print("\nModel Output Response JSON:")
            print(json.dumps(output_data, indent=4))
            print("-" * 65 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nSession terminated by user command.")
            break


if __name__ == "__main__":
    main()