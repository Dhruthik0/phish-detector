# # app.py
# import streamlit as st
# import requests

# API_URL = "http://127.0.0.1:8000/predict"

# st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")
# st.title("🛡️ Phishing URL Detector")

# url = st.text_input("Enter a URL", "http://secure-login.update-account.example.com/verify")

# if st.button("Predict") and url:
#     try:
#         response = requests.post(API_URL, json={"url": url})
#         if response.status_code == 200:
#             result = response.json()
#             st.metric("Probability (phishing)", f"{result['prob_phishing']:.3f}")
#             st.write(result)
#         else:
#             st.error(f"Error {response.status_code}: {response.text}")
#     except Exception as e:
#         st.error(f"❌ Failed to connect to API: {e}")
import streamlit as st
import pandas as pd
import joblib
import torch
from src.features import features_from_url
from src.dataset import prepare_sequences
from src.cnn_model_torch import CharCNN
import plotly.graph_objects as go

# ----------------- UI CONFIG -----------------
st.set_page_config(
    page_title="🛡️ Phishing URL Detector",
    page_icon="🔍",
    layout="centered",
)

st.title("🛡️ Phishing URL Detector")
st.markdown("### Enter a website URL below to check if it's **legit or phishing** ⚠️")

# ----------------- LOAD MODELS -----------------
@st.cache_resource
def load_models():
    # Load Random Forest model
    rf = joblib.load("models/rf_model.joblib")

    # Load PyTorch CNN model
    vocab_size = 94  # update to match your vocab size
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cnn_model = CharCNN(vocab_size=vocab_size).to(device)
    cnn_model.load_state_dict(torch.load("models/cnn_best_torch.pt", map_location=device))
    cnn_model.eval()

    return rf, cnn_model, device

rf, cnn_model, device = load_models()

# ----------------- USER INPUT -----------------
url = st.text_input("🌐 Enter URL", "http://secure-login.update-account.example.com/verify")

if st.button("🔍 Predict") and url:
    # RF prediction
    rf_feat = pd.DataFrame([features_from_url(url)])
    rf_p = float(rf.predict_proba(rf_feat)[:, 1][0])

    # CNN prediction
    X = prepare_sequences([url], max_len=200)
    X = torch.tensor(X, dtype=torch.long, device=device)
    with torch.no_grad():
        cnn_p = float(cnn_model(X).cpu().item())

    # Final probability
    prob = (rf_p + cnn_p) / 2.0

    # ----------------- GAUGE METER -----------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Phishing Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ----------------- RESULT -----------------
    if prob < 0.3:
        st.success("✅ This website looks **Safe**.")
    elif prob < 0.6:
        st.warning("⚠️ This website looks **Suspicious**. Be careful!")
    else:
        st.error("🚨 This website is likely **Phishing**! Avoid it.")

    # ----------------- EXTRA DETAILS -----------------
    with st.expander("🔎 More details"):
        st.write({
            "Final Probability": round(prob, 3),
            "RF Score": round(rf_p, 3),
            "CNN Score": round(cnn_p, 3),
            "Label": int(prob >= 0.5)
        })

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Sklearn & PyTorch")
