import streamlit as st
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

import pickle

from preprocessing import clean


# Load the trained model
with open("spam_email_models.pkl", "rb") as file:
    model = pickle.load(file)
# App title
st.title("📧 Spam Email Classifier")

st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

# Text input
email_text = st.text_area("Email Text", height=150)

# Predict button
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some text")
    else:
        prediction = model.predict([email_text])[0]

        if prediction == "spam":
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is HAM (Not Spam)")

