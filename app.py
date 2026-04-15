import streamlit as st
import pickle
import numpy as np
import requests

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🤖",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stNumberInput input {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD LOTTIE ----------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ App Settings")
st.sidebar.info("Enter feature values and click Predict")

# ---------- HEADER ----------
st.title("🚀 AI Model Deployment App")
st.markdown("### Predict outcomes using your trained Machine Learning model")

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 1])

# ---------- INPUT SECTION ----------
with col1:
    st.subheader("📥 Input Features")

    # 👉 MODIFY THESE BASED ON YOUR MODEL
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    f3 = st.number_input("Feature 3", value=0.0)
    f4 = st.number_input("Feature 4", value=0.0)

    input_data = np.array([[f1, f2, f3, f4]])

    predict_btn = st.button("🔮 Predict")

# ---------- OUTPUT SECTION ----------
with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        try:
            prediction = model.predict(input_data)

            st.success(f"✅ Prediction: {prediction[0]}")
            st.balloons()

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center><h4>✨ Built with Streamlit | AI Deployment 🚀</h4></center>",
    unsafe_allow_html=True
)
