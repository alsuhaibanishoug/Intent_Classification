import streamlit as st
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
from typing import Tuple, Dict, Optional, List
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import re


# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Intent Classification Service",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Professional Intent Classification Service\nEnterprise-Grade AI Solution"
    }
)


# ========================================
# PROFESSIONAL COLOR SCHEME
# ========================================
COLORS = {
    "primary": "#5C73B2",
    "secondary": "#E2A9F1",
    "accent": "#D75FAA",
    "highlight": "#FFDD79",
    "white": "#FFFFFF",
    "light_bg": "#F8F5FF",
    "dark_text": "#2D3748",
    "border": "#E2E8F0",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444"
}

# ========================================
# ANIMATED STYLING WITH INTERACTIVE EFFECTS
# ========================================
def apply_animated_theme():
    st.markdown(f"""
    <style>
        /* Root Variables */
        :root {{
            --primary: {COLORS['primary']};
            --secondary: {COLORS['secondary']};
            --accent: {COLORS['accent']};
            --highlight: {COLORS['highlight']};
        }}

        /* Animated Background */
        .stApp {{
            background: linear-gradient(-45deg, #F8F5FF, #F0E6FF, #FFF9E6, #F5F7FA);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }}

        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}


        h2 {{
            border-bottom: 2px solid {COLORS['secondary']};
            padding-bottom: 12px;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            border-image: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']}) 1;
            animation: borderSlide 0.8s ease-out;
        }}

        /* Animated Prediction Display */
        .prediction-display {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
            color: white;
            padding: 48px;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(92, 115, 178, 0.2);
            text-align: center;
            margin: 24px 0;
            position: relative;
            overflow: hidden;
            animation: bounceIn 0.7s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        }}


        @keyframes shimmer {{
            0% {{
                transform: translate(-100%, -100%) rotate(45deg);
            }}
            100% {{
                transform: translate(100%, 100%) rotate(45deg);
            }}
        }}

        @keyframes bounceIn {{
            0% {{
                opacity: 0;
                transform: scale(0.3) translateY(-30px);
            }}
            50% {{
                opacity: 1;
            }}
            100% {{
                transform: scale(1) translateY(0);
            }}
        }}

        .prediction-display h1 {{
            color: {COLORS['highlight']};
            font-size: 3.2rem;
            margin: 16px 0;
            letter-spacing: 1px;
            animation: textGlow 2s ease-in-out infinite;
            position: relative;
            z-index: 2;
        }}

        @keyframes textGlow {{
            0%, 100% {{text-shadow: 0 0 10px rgba(255,221,121,0.3); }}
            50% {{ text-shadow: 0 0 20px rgba(255,221,121,0.6); }}
        }}

        .prediction-display h3 {{
            color: white;
            font-size: 1.4rem;
            font-weight: 500;
            position: relative;
            z-index: 2;
        }}

        /* Animated Buttons */
        .stButton > button {{
            border-radius: 8px;
            height: 43px;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
            width: 100%;
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']}) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(92, 115, 178, 0.2) !important;
            position: relative;
            overflow: hidden;
            animation: slideInUp 0.6s ease-out;
        }}

        .stButton > button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(92, 115, 178, 0.4) !important;
        }}


        /* Animated Text Area " classification message "  */
        textarea {{
            border-radius: 10px !important;
            border: 2px solid {COLORS['border']} !important;
            background-color: white !important;
            color: {COLORS['dark_text']} !important;
            font-size: 1rem !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            transition: all 0.3s ease;
        }}


        /* Animated Progress Bar */
        .stProgress > div > div {{
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']}, {COLORS['accent']});
            background-size: 200% 100%;
            animation: progressGlow 2s ease infinite;
            border-radius: 8px;
        }}

        @keyframes progressGlow {{
            0%, 100% {{
            background-position: 0% center;
            box-shadow: 0 0 8px rgba(92, 115, 178, 0.3);

            }}
            50% {{
            background-position: 100% center;
            box-shadow: 0 0 16px rgba(92, 115, 178, 0.6);

            }}
        }}

        @keyframes gradientMove {{
            0% {{background-position: 0% center;}}
            100% {{background-position: 200% center;}}
        }}

        @keyframes slideInRight {{
            from {{
                opacity: 0;
                transform: translateX(20px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @keyframes slideInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideInDown {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .rule-card {{
            padding: 12px 8px;
            margin-bottom: 16px;
            font-size: 0.5rem;
            border-radius: 8px;
            border-left: 4px solid;
            transition: all 0.3s ease;
            animation: fadeInUp 0.4s ease-out;

        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

    </style>
    """, unsafe_allow_html=True)

apply_animated_theme()
# ========================================
# ANIMATED LOADING SPINNER
# ========================================
def animated_spinner(text: str, duration: float = 2):
    """Custom animated spinner"""
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    placeholder = st.empty()

    start_time = time.time()
    frame_index = 0

    while time.time() - start_time < duration:
        placeholder.write(f"{spinner_frames[frame_index % len(spinner_frames)]} {text}")
        frame_index += 1
        time.sleep(0.1)
    placeholder.empty()

# ========================================
# SESSION STATE INITIALIZATION
# ========================================
def init_session_state():
    """Initialize session state"""
    try:
        defaults = {
            "prediction_history": [],
            "model_loaded": False,
            "total_predictions": 0,
            "last_error": None,
            "session_start_time": datetime.now(),
            "intent_distribution": {},
            "last_text": "",
            "show_validation": True,
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    except Exception as e:
         return {"status": "error", "message": str(e)}

init_session_state()

# ==========================================
# LOAD AI MODEL
# ========================================
@st.cache_resource
def load_model() -> Tuple[Optional[object], Optional[object], bool, str]:
    """Load model with error handling"""
    try:
        with st.spinner("🚀 Initializing AI Model..."):
            MODEL_PATH = 'intent_model'

            if not Path(MODEL_PATH).exists():
                error_msg = f"Model path not found: {MODEL_PATH}"
                return None, None, False, error_msg

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model.eval()
            return model, tokenizer, True, "Success"

    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        return None, None, False, error_msg

model, tokenizer, load_success, load_message = load_model()
if not load_success:
    st.error(f"Critical Error: {load_message}")
    st.stop()

# ========================================
# INTENT LABELS
# ========================================
INTENT_LABELS = {
    0: "complaint",
    1: "general_question",
    2: "logistics",
    3: "payment",
    4: "product_inquiry",
    5: "refund",
    6: "self_harm_or_suicide_risk",
    7: "technical_support",
    8: "unknown"
}

INTENT_DESCRIPTIONS = {
    "complaint": "Customer dissatisfaction or complaint",
    "general_question": "Information inquiry",
    "logistics": "Shipping and delivery inquiry",
    "payment": "Payment and billing issue",
    "product_inquiry": "Product-related question",
    "refund": "Refund or return request",
    "self_harm_or_suicide_risk": "Critical safety alert",
    "technical_support": "Technical assistance request",
    "unknown": "insufficient information"
}

INTENT_EMOJIS = {
    "complaint": "😠",
    "general_question": "❓",
    "logistics": "📦",
    "payment": "💳",
    "product_inquiry": "🛍️",
    "refund": "↩️",
    "self_harm_or_suicide_risk": "🆘",
    "technical_support": "🔧",
    "unknown": "❓"
}

# ========================================
# EMERGENCY LEVEL SYSTEM
# ========================================
EMERGENCY_LEVELS = {
    "self_harm_or_suicide_risk": 1,
    "complaint":                 2,
    "refund":                    3,
    "payment":                   4,
    "logistics":                 5,
    "technical_support":         6,
    "product_inquiry":           7,
    "general_question":          8,
    "unknown":                   9,
}

EMERGENCY_COLORS = {
    1: "#C0392B", 2: "#E74C3C", 3: "#E67E22",
    4: "#F39C12", 5: "#F1C40F", 6: "#2ECC71",
    7: "#27AE60", 8: "#1ABC9C", 9: "#95A5A6",
}

EMERGENCY_LABELS = {
    1: "🚨 CRITICAL",   2: "🔴 HIGH",
    3: "🟠 ELEVATED",   4: "🟡 MEDIUM-HIGH",
    5: "🟡 MEDIUM",     6: "🟢 LOW-MEDIUM",
    7: "🟢 LOW",        8: "🔵 MINIMAL",
    9: "⚪ NONE",
}

# ========================================
# ENHANCED LIVE VALIDATION SYSTEM
# ========================================
class ValidationRules:
    """ input validation """
    # Constants
    MIN_LENGTH = 3
    MAX_LENGTH = 500
    @staticmethod
    def validate_text(text: str) -> Dict[str, Dict]:
        """
        Comprehensive validation with detailed feedback
        Returns: Dict with status and messages for each rule
        """
        text_stripped = text.strip() if text else ""
        text_length = len(text_stripped)

        rules = {
            "min_length": {
                "passed": text_length >= ValidationRules.MIN_LENGTH if text else False,
                "label": "Minimum Length",
                "icon": "✅" if text_length >= ValidationRules.MIN_LENGTH else "❌",
                "message": f"{text_length}/{ValidationRules.MIN_LENGTH} characters",

            },
            "has_letters": {
                "passed": bool(any(c.isalpha() for c in text)) if text else False,
                "label": "No Number only",
                "icon": "✅" if any(c.isalpha() for c in text) else "❌",
                "message": "Contains mixed characters" if any(c.isalpha() for c in text) else "Only numeric input detected"

            },
            "not_numbers_only": {
                "passed": not text_stripped.isdigit() if text else False,
                "label": "Letters Only",
                "icon": "✅" if not text_stripped.isdigit() else "❌",
                "message": "Mix of content types" if not text_stripped.isdigit() else "Only numeric characters",

            },
            "max_length": {
                "passed": text_length <= ValidationRules.MAX_LENGTH if text else True,
                "label": "Maximum Length",
                "icon": "✅" if text_length <= ValidationRules.MAX_LENGTH else "❌",
                "message": f"{text_length}/{ValidationRules.MAX_LENGTH} characters",
                "warning": text_length > ValidationRules.MAX_LENGTH * 0.85
            },
            "no_special_characters": {
              "passed": bool(text and text.isalnum() or " " in text) if text else False,
              "label": "No Symbols ",
             "icon": "✅" if (text and text.isalnum() or " " in text) else "❌",
              "message": "Letters & Numbers Only",

},
        }
        return rules

    # ──  Validation utilities for text input. ─────────────────────────────
    @staticmethod
    def is_valid(text: str) -> bool:
        """Check if all validation rules pass"""
        rules = ValidationRules.validate_text(text)
        return all(rule["passed"] for rule in rules.values())

    @staticmethod
    def get_validation_summary(text: str) -> Dict:
        """ Get overall validation status and progress """
        rules = ValidationRules.validate_text(text)
        passed_count = sum(1 for rule in rules.values() if rule["passed"])
        total_count = len(rules)

        return {
            "is_valid": all(rule["passed"] for rule in rules.values()),
            "passed_count": passed_count,
            "total_count": total_count,
            "percentage": (passed_count / total_count * 100) if total_count > 0 else 0,
            "message": f"{passed_count}/{total_count} requirements met"
        }

    # ──  Header " ✓ INPUT REQUIREMENTS  ─────────────────────────────
    @staticmethod
    def display_live_validation(text: str):
        """ Display real-time validation feedback with animations """
        rules = ValidationRules.validate_text(text)
        summary = ValidationRules.get_validation_summary(text)


        # ──  Header " ✓ INPUT REQUIREMENTS  ─────────────────────────────
        st.markdown(f"""
        <div style="
            padding: 9px 16px;
            background: linear-gradient(135deg, #F8F5FF, #F0E6FF);
            border-radius: 10px;
            border-left: 4px solid #5C73B2;
            margin-bottom: 14px;
            margin-top:15px;
            animation: slideInRight 0.5s ease-out;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="
                    font-size: 0.7rem;
                    font-weight: 700;
                    color: #5C73B2;
                    text-transform: uppercase;
                    letter-spacing: 0.9px;
                ">
                    ✓ Input Requirements
                </span>
                <span style="
                    font-size: 0.7rem;
                    font-weight: 600;
                    color: #7C3AED;
                    background: rgba(124, 58, 237, 0.1);
                    padding: 4px 10px;
                    border-radius: 10px;
                ">
                    {summary['message']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


        # ── Character counter with color coding ─────────────────────────────
        text_length = len(text.strip()) if text else 0
        if text_length == 0:
            counter_class = "neutral"
            counter_text = "0/500 characters"
            counter_color = "#94A3B8"
        elif text_length < 50:
            counter_class = "safe"
            counter_text = f"{text_length}/500 characters"
            counter_color = "#10B981"
        elif text_length < 450:
            counter_class = "safe"
            counter_text = f"{text_length}/500 characters"
            counter_color = "#10B981"
        else:
            counter_class = "warning"
            counter_text = f"{text_length}/500 characters (⚠️ Approaching limit)"
            counter_color = "#F59E0B"

        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            padding: 0 4px;
        ">
            <span style="color: {counter_color}; font-weight: 600; font-size: 0.8rem;">
                {counter_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── VALIDATION RULES GRID  ─────────────────────────────
        rule_configs = [
            {"key": "no_special_characters", "emoji_pass": "🔤", "emoji_fail": "🚫"},
            {"key": "min_length", "emoji_pass": "📏", "emoji_fail": "📏"},
            {"key": "has_letters", "emoji_pass": "🔤", "emoji_fail": "🔤"},
            {"key": "not_numbers_only", "emoji_pass": "🔢", "emoji_fail": "🔢"},
            {"key": "max_length", "emoji_pass": "📊", "emoji_fail": "⚠️"},
        ]

        col1, col2 = st.columns(2)
        for idx, config in enumerate(rule_configs):
            rule = rules[config["key"]]
            col = col1 if idx % 2 == 0 else col2

            # ── DETERMINE STYLING  ─────────────────────────────
            if not text.strip() and config["key"] != "max_length":
                status_class = "neutral"
                bg_color = "rgba(148, 163, 184, 0.05)"
                border_color = "#CBD5E1"
                text_color = "#64748B"
                status_icon = ""

            elif rule["passed"]:
                status_class = "pass"
                bg_color = "rgba(16, 185, 129, 0.08)"
                border_color = "#10B981"
                text_color = "#065F46"
                status_icon = "✅"

            else:
                status_class = "fail"
                bg_color = "rgba(239, 68, 68, 0.08)"
                border_color = "#EF4444"
                text_color = "#7F1D1D"
                status_icon = "❌"

            with col:
                st.markdown(f"""
                <div class="rule-card {status_class}" style="
                    background: {bg_color};
                    border-left-color: {border_color};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="
                                display: flex;
                                align-items: center;
                                gap: 8px;
                                margin-bottom:0px;
                            ">
                                <span style="font-size: 0.6rem;">{status_icon}</span>
                                <span style="
                                    font-size: 0.8rem;
                                    font-weight: 660;
                                    color: {text_color};
                                ">
                                    {rule['label']}
                                </span>
                            </div>
                            <div style="
                                font-size: 0.6rem;
                                color: {text_color};
                                opacity: 0.8;
                                margin-left: 5px;
                            ">
                                {rule['message']}
                            </div>
                        </div>
                """, unsafe_allow_html=True)
# ========================================
# PREDICTION ENGINE
# ========================================
class PredictionEngine:
    """ Enterprise prediction engine"""
    @staticmethod
    @st.cache_data(show_spinner=False)
    def predict(text: str) -> Dict:
        """Perform intent classification"""
        try:
            start_time = time.time()
            # ── Tokenization ─────────────────────────────
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            inputs.pop("token_type_ids", None)

            # ── Inference ────────────────────────────────
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
            if isinstance(probabilities, float):
                probabilities = [probabilities]

            # ── Build scores ─────────────────────────────
            intent_scores = [
                {
                    "intent": INTENT_LABELS[idx],
                    "confidence": round(float(prob), 4),
                    "description": INTENT_DESCRIPTIONS.get(INTENT_LABELS[idx], ""),
                    "emoji": INTENT_EMOJIS.get(INTENT_LABELS[idx], "")
                }
                for idx, prob in enumerate(probabilities)
            ]
            # ── Sort by confidence ───────────────────────
            intent_scores.sort(key=lambda x: x["confidence"], reverse=True)
            inference_time = time.time() - start_time

            # ── RETURN (NO THRESHOLD LOGIC) ──────────────
            return {
                "status": "success",
                "predicted_intent": intent_scores[0]["intent"],
                "confidence": intent_scores[0]["confidence"],
                "top_3_intents": intent_scores[:3],
                "all_intents": intent_scores,
                "inference_time_ms": round(inference_time * 1000, 2),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return { "status": "error", "message": str(e)}

# ========================================
# HEADER
# ========================================
st.markdown("""
<div style='text-align: center; padding: 40px 20px; margin-bottom: 40px;'>
    <h1 style='font-size: 7.8rem; margin: 0; color:#2D3748'> Fahim </h1>
    <p style='font-size: 1.1rem; color:#5C73B2; margin-top: 12px; animation: fadeInDown 0.8s ease-out 0.3s both;'>
        Customer Service Intent Classification Model </p> </div>
""", unsafe_allow_html=True)
st.markdown("---")

# ========================================
# MAIN INPUT SECTION WITH LIVE VALIDATION
# ========================================

#if "show_validation" not in st.session_state:
   # st.session_state.show_validation = True
#if "last_text" not in st.session_state:
    #st.session_state.last_text = ""

st.markdown("<h2> Classification Request</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([2.5, 1.3])
with col1:
    user_text = st.text_area(" Enter your message here",
        key="user_text",
        height=150,
        placeholder=" Provide the customer message or query for intent classification...")
    is_input_valid = ValidationRules.is_valid(user_text)
    if user_text != st.session_state.last_text:
      st.session_state.last_text = user_text
      st.session_state.show_validation = True
with col2:
    # ── Cancel validation rules only after all conditions have been verified ─────────────────────────────
    if st.session_state.show_validation and not is_input_valid:
        ValidationRules.display_live_validation(user_text or "")

st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)


# ── Action buttons ─────────────────────────────
btn_col2, btn_col3 = st.columns(2)
with btn_col2:
        submit_btn = st.button(
        "🚀 Classify",
        type="primary",
        use_container_width=True,
    )

with btn_col3:
    clear_btn = st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=lambda: st.session_state.update(
    {
        "user_text": "",
        "show_validation": True,
        "last_text": ""
    }))

# ========================================
# PREDICTION PROCESSING WITH ANIMATIONS
# ========================================
if submit_btn:

    # ── Check empty  ─────────────────────────────
    if not user_text or not user_text.strip():
        st.error("Please provide text for classification")
        st.stop()

    if not ValidationRules.is_valid(user_text):
        st.error("Please satisty all requirements before classification ")
        st.stop()

    st.session_state.show_validation = False
    try:
            prediction = PredictionEngine.predict(user_text)
            if prediction["status"] == "error":
                st.error(f"Classification Error: {prediction['message']}")
            else:
                st.session_state.total_predictions += 1
                st.session_state.prediction_history.append({
                    "text": user_text,
                    "intent": prediction["predicted_intent"],
                    "confidence": prediction["confidence"],
                    "timestamp": datetime.now()
                })

                intent = prediction["predicted_intent"]
                if intent not in st.session_state.intent_distribution:
                    st.session_state.intent_distribution[intent] = 0
                st.session_state.intent_distribution[intent] += 1
                st.markdown("---")



                # ── PRIMARY RESULT WITH ANIMATION AND EMERGENCY LEVEL  ─────────────────────────────
                emoji = INTENT_EMOJIS.get(prediction['predicted_intent'], "")
                emergency_level = EMERGENCY_LEVELS.get(prediction['predicted_intent'], 9)
                emergency_color = EMERGENCY_COLORS.get(emergency_level, "#95A5A6")
                emergency_label = EMERGENCY_LABELS.get(emergency_level, "⚪ NONE")

                st.markdown(f"""
                <div class='prediction-display animate-in' style='border: 3px solid {emergency_color};'>
                    <h3>Classification Result</h3>
                    <h1>{emoji} {prediction['predicted_intent'].replace('_', ' ').upper()}</h1>

                """, unsafe_allow_html=True)

                 # ── DETAILED METRICS   ─────────────────────────────
                st.markdown("<h2> Classification Details </h2>", unsafe_allow_html=True)
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Primary Classification", prediction["predicted_intent"].replace("_", " ").title())

                with metric_col2:
                    st.metric("Confidence Level", f"{prediction['confidence']:.2%}" )

                with metric_col3:
                    st.metric( "Priority Level", emergency_label)
                st.markdown("---")

                # ── TOP PREDICTIONS WITH ANIMATION ─────────────────────────────
                st.markdown("<h2>Top 3 Candidate Intents</h2>", unsafe_allow_html=True)
                for rank, item in enumerate(prediction['top_3_intents'], 1):
                    with st.container():
                        col_rank, col_info, col_score = st.columns([0.8, 2.2, 1])

                        with col_rank:
                            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                            st.markdown(f"<p style='font-size: 2.9rem; margin: 0; animation: bounce 0.6s ease-out {0.1 * rank}s both;'>{medal}</p>", unsafe_allow_html=True)

                        with col_info:
                            st.markdown(f"**{item['intent'].replace('_', ' ').title()}**")
                            st.caption(item['description'])

                        with col_score:
                            st.markdown(f"<p style='font-weight: 700; font-size: 1.9rem;'>{item['confidence']:.2%}</p>", unsafe_allow_html=True)
                        st.progress(item['confidence'])

                st.markdown("---")

                # Success Animation
                st.balloons()

    except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if clear_btn:
    st.rerun()

# ========================================
# ANIMATED FOOTER
# ========================================
st.markdown(f"""
<style>
    /* Fixed Footer Styling */
    .fixed-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #E2E8F0;
        padding: 8px 20px;
        text-align: center;
        font-size: 11px;
        opacity: 0.95;
        z-index: 100;
        box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.05);
    }}

    .stApp {{padding-bottom: 30px;}}
</style>
<div class='fixed-footer'>
    <p style='margin: 0; opacity: 0.7;'>© 2026 | Power by Fahiam Team </p>
</div>
""", unsafe_allow_html=True)
