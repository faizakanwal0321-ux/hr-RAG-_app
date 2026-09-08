import streamlit as st

st.title("📄 HR Policy Assistant")

st.write("Upload an HR policy PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload your HR Policy PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("PDF uploaded successfully!")
