import streamlit as st
import joblib
import os
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import PyPDF2

MODEL_PATH = "content_inspector_combo.pkl"

@st.cache_resource(show_spinner=False)
def load_lr_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)['pipeline']

@st.cache_resource(show_spinner=False)
def load_transformer():
    return pipeline("text-classification", model="roberta-base-openai-detector")

def detect_language(text):
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "fa"
    else:
        return "en"

def split_paragraphs(text):
    return [p for p in text.split('\n') if len(p.strip()) > 30]

def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

st.set_page_config(page_title="GhostWriterDetector", page_icon="👻", layout="centered")

theme = st.toggle("🌗 Night mode", value=True)
css = """
.stApp {background-color: #181925;}
.big-title {color: #f7d354; font-size: 3rem; font-weight: bold; letter-spacing: 2px; text-shadow: 0 4px 32px #0002; margin-bottom: 10px; text-align:center;}
.subtitle {color: #b8b8b8; font-size: 1.2rem; text-align: center; margin-bottom: 40px;}
.result-human {background: linear-gradient(90deg,#1faa6c,#00d8c4); border-radius: 12px; color: #fff; font-size: 1.4rem; font-weight: bold; text-align: center; margin-top:20px; box-shadow: 0 2px 12px #0002; padding: 1.1rem 0.5rem 1.1rem 0.5rem;}
.result-ai {background: linear-gradient(90deg,#e53d3e,#e84e7c); border-radius: 12px; color: #fff; font-size: 1.4rem; font-weight: bold; text-align: center; margin-top:20px; box-shadow: 0 2px 12px #0002; padding: 1.1rem 0.5rem 1.1rem 0.5rem;}
"""
css_light = css.replace("#181925", "#fafafd").replace("#f7d354", "#b09c33")
st.markdown(f"<style>{css if theme else css_light}</style>", unsafe_allow_html=True)

st.markdown('<div class="big-title">👻 GhostWriterDetector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'AI vs. Human Text Detection | Paste or upload .txt/.pdf (English or فارسی) | Feedback driven improvement'
    '</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Paste Text", "Upload File"])
with tab1:
    user_text = st.text_area("Paste your text here (English or فارسی)", height=160, key="input_text")
with tab2:
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            user_text = extract_pdf_text(uploaded_file)
        else:
            user_text = StringIO(uploaded_file.read().decode("utf-8")).read()

segments = split_paragraphs(user_text) if len(user_text) > 0 else []

st.divider()

if st.button("Detect", use_container_width=True, type="primary"):
    if not user_text.strip():
        st.warning("Please paste text or upload a file.")
    else:
        lr_model = load_lr_model()
        transformer = load_transformer()
        res_table = []
        for idx, seg in enumerate(segments if segments else [user_text]):
            lang = detect_language(seg)
            if lang == "fa":
                st.info(f"Segment {idx+1} detected as فارسی. Detection is less accurate.")
            if lr_model:
                proba = lr_model.predict_proba([seg])[0]
                lr_label = lr_model.predict([seg])[0]
                lr_conf = proba[lr_label]
            else:
                lr_label, lr_conf = None, None
            result = transformer(seg)[0]
            tr_label = 0 if result['label'].lower() == 'human' else 1
            tr_conf = result['score']
            votes = [lr_label, tr_label]
            ensemble_label = 1 if sum(votes) > 1 else 0
            avg_conf = (lr_conf + tr_conf)/2 if lr_conf is not None else tr_conf
            res_table.append({"Segment": idx+1, "LR": lr_label, "LR_Conf": lr_conf, "TR": tr_label, "TR_Conf": tr_conf, "Ensemble": ensemble_label, "Conf": avg_conf, "Text": seg})

        for r in res_table:
            label = r["Ensemble"]
            conf = r["Conf"]
            detail = f"""
            <div style='text-align:center;margin-top:10px'>
            <b>LogReg:</b> {'AI' if r['LR'] else 'Human'} ({r['LR_Conf']*100:.1f}%)<br>
            <b>Transformer:</b> {'AI' if r['TR'] else 'Human'} ({r['TR_Conf']*100:.1f}%)<br>
            <b>Ensemble:</b> <b>{'AI' if label else 'Human'}</b>
            </div>
            """
            if label == 0:
                st.markdown(
                    f'<div class="result-human">🧑‍💼 <b>Human-written</b> <br> <span style="font-size:1.1rem;">Confidence: <b>{conf*100:.1f}%</b></span>{detail}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-ai">🤖 <b>AI-generated</b> <br> <span style="font-size:1.1rem;">Confidence: <b>{conf*100:.1f}%</b></span>{detail}</div>',
                    unsafe_allow_html=True
                )
        labels = [r["Ensemble"] for r in res_table]
        st.subheader("Voting & Confidence Summary")
        fig, ax = plt.subplots()
        ax.pie([labels.count(0), labels.count(1)], labels=['Human', 'AI'], autopct='%1.1f%%', colors=["#1faa6c", "#e84e7c"])
        st.pyplot(fig)
        st.progress(int(res_table[0]['Conf']*100) if res_table else 0)

        st.subheader("Your Feedback")
        feedback = st.radio("Is this prediction correct?", ("Yes", "No"), horizontal=True, key="fb")
        if st.button("Submit Feedback"):
            if "feedback.csv" not in os.listdir():
                pd.DataFrame(columns=["Text", "Model_Human_Or_AI", "User_Confirm"]).to_csv("feedback.csv", index=False)
            df = pd.read_csv("feedback.csv")
            df.loc[len(df)] = [user_text, "AI" if labels[0] else "Human", feedback]
            df.to_csv("feedback.csv", index=False)
            st.success("Thanks for your feedback! ❤️")

st.markdown("""
    <div style="margin-top:32px; text-align:center;">
        <a href="https://github.com/Canyildiz1386/GhostWriterDetector" target="_blank" style="color:#b8b8b8; font-size:0.95rem; text-decoration:none;">
        <span style="font-size:1.2rem;">🌐</span> GitHub: GhostWriterDetector
        </a>
        <br>
        <span style="color:#393e46; font-size:0.92rem;">Made with 💛 by Can Yildiz</span>
    </div>
""", unsafe_allow_html=True)
