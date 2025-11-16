import streamlit as st
import requests
import pandas as pd
from typing import Dict
import os

API_BASE_URL = os.getenv("API_BASE_URL", "https://predict-income-latest.onrender.com")
API_URL = f"{API_BASE_URL}/predict"
INFO_URL = f"{API_BASE_URL}/info"

st.set_page_config(page_title="Income Prediction", page_icon="💼", layout="wide")

st.title("Income >50K Prediction")
st.write("Fill the form (left) and click Predict. The app calls the FastAPI service and shows probability + decision.")

# --- helper: load model info (threshold, features) from API ---
@st.cache_data(ttl=300)
def load_model_info() -> Dict:
    try:
        r = requests.get(INFO_URL, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

meta = load_model_info()
THRESHOLD = meta.get("threshold", None)
FEATURE_OPTIONS = meta.get("feature_options", {})

# --- simple education -> education_num mapping (common mapping used by Adult dataset) ---
EDU_TO_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-acdm": 11,
    "Assoc-voc": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

# --- Fallback options (if API is down) ---
WORKCLASS_OPTIONS = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
EDUCATION_OPTIONS = list(EDU_TO_NUM.keys())
MARITAL_OPTIONS = [
    "Married-civ-spouse", "Divorced", "Never-married", 
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
]
OCCUPATION_OPTIONS = [
    "Tech-support", "Craft-repair", "Other-service", "Sales", 
    "Exec-managerial", "Prof-specialty", "Protective-serv", 
    "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", 
    "Farming-fishing", "Armed-Forces", "Priv-house-serv", "Adm-clerical"
]
RELATIONSHIP_OPTIONS = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
RACE_OPTIONS = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
SEX_OPTIONS = ["Male", "Female"]
COUNTRY_OPTIONS = [
    'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
    'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
    'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
    'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
    'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
    'China', 'Japan', 'Yugoslavia', 'Peru',
    'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
    'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
    'Holand-Netherlands'
]

# --- Sidebar form ---
with st.sidebar.form("input_form"):
    st.header("Input features")
    age = st.number_input("Age", min_value=15, max_value=100, value=37)
    workclass = st.selectbox("Workclass", FEATURE_OPTIONS.get("workclass", WORKCLASS_OPTIONS), index=0)
    education = st.selectbox("Education", FEATURE_OPTIONS.get("education", EDUCATION_OPTIONS), index=EDUCATION_OPTIONS.index("HS-grad")) # Default to HS-grad
    education_num = EDU_TO_NUM[education]
    marital_status = st.selectbox("Marital status", FEATURE_OPTIONS.get("marital_status", MARITAL_OPTIONS), index=0)
    occupation = st.selectbox("Occupation", FEATURE_OPTIONS.get("occupation", OCCUPATION_OPTIONS), index=4)
    relationship = st.selectbox("Relationship", FEATURE_OPTIONS.get("relationship", RELATIONSHIP_OPTIONS), index=2)
    race = st.selectbox("Race", FEATURE_OPTIONS.get("race", RACE_OPTIONS), index=0)
    sex = st.selectbox("Sex", FEATURE_OPTIONS.get("sex", SEX_OPTIONS), index=0)
    capital_gain = st.number_input("Capital gain", min_value=0, value=100000, step=1)
    capital_loss = st.number_input("Capital loss", min_value=0, value=0, step=1)
    hours_per_week = st.number_input("Hours per week", min_value=0, max_value=168, value=40)
    native_country = st.selectbox("Native country", FEATURE_OPTIONS.get("native_country", COUNTRY_OPTIONS), index=0)

    col_reset, col_predict = st.columns(2)
    with col_reset:
        reset_button = st.form_submit_button("Reset Form")
    with col_predict:
        submitted = st.form_submit_button("Predict", type="primary")

if reset_button:
    st.session_state["reset_form"] = True
    st.rerun()

if "reset_form" in st.session_state and st.session_state["reset_form"]:
    st.session_state["reset_form"] = False
    st.rerun()

# --- Main content / results ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Model info")
    if meta:
        st.markdown(f"- Threshold used by model: **{meta.get('threshold', 'N/A')}**")
        st.markdown(f"- Model: **{meta.get('model', 'unknown')}**")
        if meta.get("features"):
            st.markdown(f"- Features: {', '.join(meta['features'][:8])} ...")
        if st.checkbox("Show raw model metadata"):
            st.json(meta)
    else:
        st.info("No model metadata found at /info. Using local defaults.")

with col2:
    if submitted:
        # Client-side validation
        if not (16 <= age <= 100):
            st.error("Age must be between 16 and 100.")
            st.stop()
        
        payload = {
            "age": int(age),
            "workclass": workclass,
            "education": education,
            "education_num": int(education_num),
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": int(capital_gain),
            "capital_loss": int(capital_loss),
            "hours_per_week": int(hours_per_week),
            "native_country": native_country
        }

        # Custom spinner (waveform)
        spinner_html = """
        <style>
        .loader-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 20px;
        }
        .waveform {
          --bar-width: 4px;
          --bar-height: 20px;
          --bar-spacing: 3px;
          --bar-color: #09f;
          display: flex;
          justify-content: center;
          align-items: flex-end;
          height: var(--bar-height);
        }
        .waveform__bar {
          width: var(--bar-width);
          height: 100%;
          background-color: var(--bar-color);
          margin: 0 var(--bar-spacing);
          animation: waveform-animation 1.2s ease-in-out infinite;
        }
        .waveform__bar:nth-child(1) { animation-delay: 0s; }
        .waveform__bar:nth-child(2) { animation-delay: 0.2s; }
        .waveform__bar:nth-child(3) { animation-delay: 0.4s; }
        .waveform__bar:nth-child(4) { animation-delay: 0.6s; }

        @keyframes waveform-animation {
          0% { transform: scaleY(0.1); }
          50% { transform: scaleY(1); }
          100% { transform: scaleY(0.1); }
        }
        .loader-text {
          margin-top: 15px;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
          color: #555;
          font-size: 1.1em;
        }
        </style>
        <div class="loader-container">
          <div class="waveform">
            <div class="waveform__bar"></div>
            <div class="waveform__bar"></div>
            <div class="waveform__bar"></div>
            <div class="waveform__bar"></div>
          </div>
          <div class="loader-text">Analyzing...</div>
        </div>
        """
        
        placeholder = st.empty()
        placeholder.markdown(spinner_html, unsafe_allow_html=True)

        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            result = r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {e}")
            result = None
        finally:
            placeholder.empty() # Clear the spinner

        if result:
            prob = result.get("probability_over_50k")
            pred = result.get("prediction")
            thresh = result.get("threshold", THRESHOLD)
            st.metric("Prediction", pred, delta=f"threshold {thresh}")
            st.subheader("Probability")
            pct = int(round(prob * 100))
            st.progress(pct)
            color = "green" if prob >= (thresh or 0.5) else "red"
            st.markdown(f"<div style='font-size:20px;color:{color}'>P(>50K) = **{prob:.4f}**</div>", unsafe_allow_html=True)

            with st.expander("Raw response"):
                st.json(result)

            if st.checkbox("Save request to CSV"):
                log_file = "predictions_log.csv"
                file_exists = os.path.exists(log_file)
                df = pd.DataFrame([payload])
                df["probability_over_50k"] = prob
                df.to_csv(log_file, mode="a", header=not file_exists, index=False)
                st.success(f"Saved to {log_file}")
            
            with st.expander("Submitted Input Data"):
                st.json(payload)

    else:
        st.info("Fill the form and click Predict")

# footer
st.markdown("---")
st.caption("Streamlit UI for Adult Income prediction. Update API_URL if FastAPI is deployed elsewhere.")
