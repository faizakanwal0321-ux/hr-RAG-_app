````markdown
# HR Policy Assistant

A beginner-friendly RAG application that allows users to upload an HR policy PDF and ask questions about it.

## Features

- Upload an HR policy PDF
- Extract text from the PDF
- Split text into chunks
- Create embeddings using Sentence Transformers
- Search relevant information using FAISS
- Generate answers using Groq
- Simple Streamlit interface

## Technologies

- Python
- Streamlit
- PyMuPDF
- FAISS
- Sentence Transformers
- Groq
- NumPy

## How It Works

```text
HR Policy PDF
      ↓
   PyMuPDF
      ↓
 Extract Text
      ↓
 Text Chunks
      ↓
Sentence Transformers
      ↓
 Embeddings
      ↓
    FAISS
      ↓
Relevant Information
      ↓
     Groq
      ↓
    Answer
````

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Example Questions

After uploading an HR policy PDF, you can ask:

* How many annual leave days are available?
* What are the working hours?
* What is the sick leave policy?
* Can employees work remotely?
* What is the resignation notice period?

## Learning Goals

This project was created to learn:

* RAG
* Embeddings
* Vector search
* FAISS
* LLMs
* Streamlit
* Python
* GitHub

## Author

Faiza Kanwal

```
```
