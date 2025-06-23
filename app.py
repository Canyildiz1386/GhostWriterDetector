import streamlit as st
import joblib
import os

MODEL_PATH = "content_inspector.pkl"

st.set_page_config(page_title="GhostWriterDetector", page_icon="👻", layout="centered")

st.markdown("""
    <style>
    .main {background: #181925;}
    .stApp {background-color: #181925;}
    .big-title {
        color: #f7d354;
        font-size: 3rem;
        font-weight: bold;
        letter-spacing: 2px;
        text-shadow: 0 4px 32px #0002;
        margin-bottom: 10px;
        text-align:center;
    }
    .subtitle {
        color: #b8b8b8;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 40px;
    }
    .result-human {
        background: linear-gradient(90deg,#1faa6c,#00d8c4);
        border-radius: 12px;
        color: #fff;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-top:20px;
        box-shadow: 0 2px 12px #0002;
        padding: 1.1rem 0.5rem 1.1rem 0.5rem;
    }
    .result-ai {
        background: linear-gradient(90deg,#e53d3e,#e84e7c);
        border-radius: 12px;
        color: #fff;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-top:20px;
        box-shadow: 0 2px 12px #0002;
        padding: 1.1rem 0.5rem 1.1rem 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">👻 GhostWriterDetector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Detect if your text is <span style="color:#1faa6c">human</span> or <span style="color:#e53d3e">AI</span>-generated.<br> Powered by advanced ML and real-world datasets.'
    '</div>',
    unsafe_allow_html=True
)

user_text = st.text_area(
    "Paste your text here",
    height=180,
    placeholder="Type or paste any text you want to check...",
    key="input_text"
)

def predict(text):
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Please run `python model_gen.py --train` first.")
        return None, None
    model_data = joblib.load(MODEL_PATH)
    pipe = model_data['pipeline']
    proba = pipe.predict_proba([text])[0]
    label = pipe.predict([text])[0]
    conf = proba[label]
    return label, conf

if st.button("Detect", use_container_width=True, type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        label, conf = predict(user_text)
        if label is not None:
            if label == 0:
                st.markdown(
                    f'<div class="result-human">🧑‍💼 <b>Human-written</b> <br> <span style="font-size:1.1rem;">Confidence: <b>{conf*100:.1f}%</b></span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-ai">🤖 <b>AI-generated</b> <br> <span style="font-size:1.1rem;">Confidence: <b>{conf*100:.1f}%</b></span></div>',
                    unsafe_allow_html=True
                )

st.markdown("""
    <div style="margin-top:48px; text-align:center;">
        <a href="https://github.com/yourusername/GhostWriterDetector" target="_blank" style="color:#b8b8b8; font-size:0.95rem; text-decoration:none;">
        <span style="font-size:1.2rem;">🌐</span> GitHub: GhostWriterDetector
        </a>
        <br>
        <span style="color:#393e46; font-size:0.92rem;">Made with 💛 by John</span>
    </div>
""", unsafe_allow_html=True)
