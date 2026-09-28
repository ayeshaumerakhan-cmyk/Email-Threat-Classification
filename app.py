"""Streamlit prototype for email threat classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

from src.email_threat.dataset import parse_message
from src.email_threat.features import FEATURE_NAMES, extract_features, feature_frame

MODEL_PATH = Path(__file__).resolve().parent / "models" / "email_threat_model.joblib"

st.set_page_config(page_title="Email Threat Classification", page_icon="✉️")
st.title("Email Threat Classification")
st.caption("Educational spam/ham prediction demo — not a production security tool.")

if not MODEL_PATH.is_file():
    st.error("The model has not been trained yet. Run `python -m src.email_threat.dataset` and `python -m src.email_threat.train`.")
    st.stop()

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
if model_bundle.get("feature_names") != FEATURE_NAMES:
    st.error("The model's feature schema does not match this app. Retrain the model.")
    st.stop()

st.warning(
    "This model was trained on historical 2002–2003 email. It can be wrong; "
    "never use its result alone to trust, block, or quarantine a real message."
)

uploaded = st.file_uploader("Optional: analyze an .eml file", type=["eml"])
if uploaded is not None:
    subject, body, sender, attachment_count = parse_message(uploaded.getvalue())
    st.info(f"Loaded email from {sender or 'unknown sender'}")
else:
    with st.form("email_input"):
        sender = st.text_input("Sender", placeholder="sender@example.com")
        subject = st.text_input("Subject")
        body = st.text_area("Message body", height=220)
        attachment_count = st.number_input(
            "Number of attachments", min_value=0, max_value=50, value=0, step=1
        )
        submitted = st.form_submit_button("Classify email")

    if not submitted:
        st.stop()

if uploaded is not None or submitted:
    features = extract_features(
        subject=subject,
        body=body,
        sender=sender,
        attachment_count=int(attachment_count),
    )
    row = feature_frame(features)
    probability = float(model.predict_proba(row)[0, 1])
    label = int(model.predict(row)[0])
    if label == 1:
        st.error(f"Model prediction: Spam (estimated probability {probability:.1%})")
    else:
        st.success(f"Model prediction: Ham / not spam (estimated spam probability {probability:.1%})")
    st.subheader("Extracted indicators")
    st.table(row.T.rename(columns={0: "Value"}))
    st.caption("Probability is a model score, not a calibrated guarantee or security verdict.")
