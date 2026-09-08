````markdown
# HR Policy Assistant using RAG

A beginner-friendly RAG (Retrieval-Augmented Generation) application that allows users to upload an HR policy PDF and ask questions about its content.

## Features

- Upload an HR policy PDF
- Extract text from PDF
- Split text into chunks
- Generate embeddings using Sentence Transformers
- Store embeddings using FAISS
- Retrieve relevant information from the document
- Generate answers using Groq
- Simple and interactive Streamlit interface

## Technologies Used

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
Retrieve Relevant Information
      ↓
     Groq
      ↓
    Answer
````

## Project Structure

```text
hr-policy-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── sample_hr_policy.txt
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/hr-policy-rag.git
```

### 2. Go to the project folder

```bash
cd hr-policy-rag
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For Mac/Linux:

```bash
source venv/bin/activate
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

## Groq API Key

This project uses Groq to generate answers.

You need a Groq API key to run the application.

The API key is entered directly in the Streamlit sidebar.

Do not upload your API key to GitHub.

## Run the Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser.

## Example Questions

After uploading an HR policy PDF, you can ask questions such as:

```text
How many annual leave days are employees entitled to?

What are the standard working hours?

How many sick leave days are available?

Can employees work remotely?

What is the resignation notice period?
```

## RAG Process

The application uses RAG to answer questions from the uploaded HR policy.

### 1. Retrieval

The user's question is converted into an embedding.

FAISS searches for the most relevant text chunks from the uploaded PDF.

### 2. Generation

The retrieved text is sent to the Groq language model along with the user's question.

Groq generates an answer based on the retrieved information.

This helps the application answer questions using the uploaded HR policy instead of relying only on the model's general knowledge.

## What I Learned

Through this project, I learned the basics of:

* RAG applications
* PDF text extraction
* Text chunking
* Embeddings
* Vector similarity search
* FAISS
* Sentence Transformers
* LLM APIs
* Groq
* Streamlit
* Python
* Git and GitHub

## Future Improvements

* Support multiple PDF files
* Add chat history
* Show page numbers for answers
* Improve text chunking
* Store FAISS indexes
* Add authentication
* Use environment variables for API keys
* Deploy the application online

## Author

Faiza Kanwal

Built as a learning project to understand RAG and AI application development.

```
```
