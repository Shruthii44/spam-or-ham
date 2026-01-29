import streamlit as st
import os
import pickle
import nltk
from preprocessing import clean

# ------------------ NLTK Cloud-safe setup ------------------
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

for pkg in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_PATH)

# ------------------ Load model ------------------
MODEL_PATH = os.path.join(os.getcwd(), "spam_email_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ------------------ Streamlit UI ------------------
st.title("📧 Spam Email Classifier")
st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

email_text = st.text_area("Email Text", height=150)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        prediction = model.predict([email_text])[0]
        if str(prediction).lower() == "spam":
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is HAM (Not Spam)")




