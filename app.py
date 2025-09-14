import streamlit as st
import pandas as pd
import joblib
import torch
from src.features import features_from_url
from src.dataset import prepare_sequences
from src.cnn_model_torch import CharCNN
import plotly.graph_objects as go


st.set_page_config(
    page_title="🛡️ Phishing URL Detector",
    page_icon="🔍",
    layout="centered",
)

st.title("🛡️ Phishing URL Detector")
st.markdown("### Enter a website URL below to check if it's **legit or phishing** ⚠️")


@st.cache_resource
def load_models():
    
    rf = joblib.load("models/rf_model.joblib")

   
    vocab_size = 94  
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cnn_model = CharCNN(vocab_size=vocab_size).to(device)
    cnn_model.load_state_dict(torch.load("models/cnn_best_torch.pt", map_location=device))
    cnn_model.eval()

    return rf, cnn_model, device

rf, cnn_model, device = load_models()


url = st.text_input("🌐 Enter URL", "http://secure-login.update-account.example.com/verify")

if st.button("🔍 Predict") and url:
   
    rf_feat = pd.DataFrame([features_from_url(url)])
    rf_p = float(rf.predict_proba(rf_feat)[:, 1][0])

   
    X = prepare_sequences([url], max_len=200)
    X = torch.tensor(X, dtype=torch.long, device=device)
    with torch.no_grad():
        cnn_p = float(cnn_model(X).cpu().item())

  
    prob = (rf_p + cnn_p) / 2.0

 
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

   
    if prob < 0.3:
        st.success("✅ This website looks **Safe**.")
    elif prob < 0.6:
        st.warning("⚠️ This website looks **Suspicious**. Be careful!")
    else:
        st.error("🚨 This website is likely **Phishing**! Avoid it.")

    
    with st.expander("🔎 More details"):
        st.write({
            "Final Probability": round(prob, 3),
            "RF Score": round(rf_p, 3),
            "CNN Score": round(cnn_p, 3),
            "Label": int(prob >= 0.5)
        })

st.markdown("---")
st.caption("Made with ❤️ 🧠 using Streamlit, Sklearn & PyTorch")
