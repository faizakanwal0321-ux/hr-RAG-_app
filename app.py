import streamlit as st
from groq import Groq

st.title("📄 HR Policy Assistant")

st.write("Upload an HR policy PDF and ask questions about it.")

# API Key
api_key = st.sidebar.text_input(
    "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",
    type="password"
)

if api_key:
    client = Groq(api_key=api_key)
    st.success("API key added!")

uploaded_file = st.file_uploader(
    "Upload your HR Policy PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("PDF uploaded successfully!")

    question = st.text_input(
        "Ask a question about the HR policy:"
    )

    if question:
        st.write("Your question:", question)
