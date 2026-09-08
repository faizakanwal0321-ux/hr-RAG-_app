```python
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 HR Policy Assistant")
st.write("Upload an HR policy PDF and ask questions about it.")


# -----------------------------
# Groq API Key
# -----------------------------
st.sidebar.header("Settings")

groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar.")
    st.stop()


# Create Groq client
client = Groq(api_key=groq_api_key)


# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Extract Text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_file):

    text = ""

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


# -----------------------------
# Split Text into Chunks
# -----------------------------
def create_chunks(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# Create FAISS Index
# -----------------------------
def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Search Relevant Chunks
# -----------------------------
def search_chunks(question, chunks, index, top_k=3):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    relevant_chunks = []

    for i in indices[0]:

        if i < len(chunks):
            relevant_chunks.append(chunks[i])

    return relevant_chunks


# -----------------------------
# Ask Groq
# -----------------------------
def ask_groq(question, context):

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question using ONLY the information
provided in the HR policy context below.

If the answer cannot be found in the context,
say:

"I could not find this information in the uploaded HR policy."

Do not make up information.

HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload HR Policy PDF",
    type=["pdf"]
)


# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file:

    with st.spinner("Reading PDF..."):

        text = extract_text_from_pdf(uploaded_file)

    if not text.strip():

        st.error("Could not extract text from this PDF.")

    else:

        st.success("PDF successfully read!")

        # Create chunks
        chunks = create_chunks(text)

        st.write(f"📚 Created {len(chunks)} text chunks.")

        # Create FAISS index
        with st.spinner("Creating search index..."):

            index, embeddings = create_faiss_index(chunks)

        st.success("HR policy is ready for questions!")

        # -----------------------------
        # Ask Question
        # -----------------------------
        question = st.text_input(
            "Ask a question about the HR policy:"
        )

        if question:

            with st.spinner("Searching HR policy..."):

                relevant_chunks = search_chunks(
                    question,
                    chunks,
                    index
                )

            context = "\n\n".join(relevant_chunks)

            with st.spinner("Generating answer..."):

                answer = ask_groq(
                    question,
                    context
                )

            st.subheader("💬 Answer")

            st.write(answer)

            # Show retrieved information
            with st.expander("🔎 View retrieved policy sections"):

                for i, chunk in enumerate(relevant_chunks):

                    st.write(f"**Section {i + 1}**")

                    st.write(chunk)

                    st.divider()
```

### What you are learning here

Don't try to memorize the whole code. Understand these five pieces:

| Part        | Technology            | What it does               |
| ----------- | --------------------- | -------------------------- |
| PDF reading | PyMuPDF               | Extracts text from PDF     |
| Embeddings  | Sentence Transformers | Converts text into numbers |
| Search      | FAISS                 | Finds relevant text        |
| LLM         | Groq                  | Generates the answer       |
| Interface   | Streamlit             | Creates the web app        |

---

# 2. `requirements.txt`

```text
streamlit
pymupdf
faiss-cpu
sentence-transformers
groq
numpy
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

# 3. `sample_hr_policy.txt`

You can use this to create your own sample PDF for testing.

```text
COMPANY HR POLICY

1. Annual Leave

Employees are entitled to 20 days of annual paid leave per year.

Employees should submit their leave request at least 5 working days before
the requested leave date.

Leave requests must be approved by the employee's manager.

2. Sick Leave

Employees can take up to 10 days of paid sick leave per year.

Employees should inform their manager as soon as possible when they are
unable to come to work because of illness.

A medical certificate may be required for extended sick leave.

3. Working Hours

The standard working hours are 9:00 AM to 5:00 PM, Monday to Friday.

Employees receive a one-hour lunch break.

4. Remote Work

Employees may request to work remotely up to two days per week.

Remote work must be approved by the employee's manager.

5. Parental Leave

Eligible employees can request parental leave according to company policy
and applicable employment regulations.

Employees should contact the HR department for information about their
specific parental leave entitlement.

6. Employee Conduct

Employees are expected to behave professionally and respectfully toward
colleagues, customers, and management.

Harassment, discrimination, bullying, and threatening behavior are not
permitted in the workplace.

7. Resignation

Employees who wish to resign should submit a written resignation notice
to their manager and HR department.

The standard notice period is 30 days.

8. Performance Reviews

Employees receive a formal performance review once every year.

Managers use performance reviews to discuss employee performance, goals,
development opportunities, and areas for improvement.

9. Training

Employees may participate in company training programs to improve their
professional and technical skills.

Employees should discuss relevant training opportunities with their manager.

10. HR Support

Employees can contact the HR department for questions about company
policies, benefits, leave, workplace concerns, and other employment matters.
```

---

# 4. `.gitignore`

This is important because you **should not upload your API key to GitHub**.

```text
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
venv/
.venv/
env/

# Environment variables
.env

# IDE files
.vscode/
.idea/

# Operating system
.DS_Store
Thumbs.db

# Streamlit
.streamlit/

# Generated files
*.log
```

---

# 5. `README.md`

````markdown
# 📄 HR Policy Assistant using RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) application
that allows users to upload an HR policy PDF and ask questions about
the information contained in the document.

The application retrieves relevant information from the uploaded PDF
and uses an LLM to generate an answer based on that information.

## 🚀 Features

- Upload an HR policy PDF
- Extract text from the PDF
- Split the document into smaller chunks
- Create text embeddings
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Ask questions about the HR policy
- Generate answers using Groq
- Display the retrieved sections used for the answer
- Simple Streamlit interface

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq
- NumPy

## 🧠 How RAG Works

The application follows these steps:

1. User uploads an HR policy PDF.
2. PyMuPDF extracts text from the PDF.
3. The text is divided into smaller chunks.
4. Sentence Transformers converts each chunk into an embedding.
5. FAISS stores the embeddings for similarity search.
6. User asks a question.
7. The question is converted into an embedding.
8. FAISS finds the most relevant policy chunks.
9. The retrieved chunks are given to the Groq LLM.
10. Groq generates an answer using the retrieved context.

### RAG Architecture

```text
             HR Policy PDF
                   |
                   ↓
              PyMuPDF
                   |
                   ↓
            Extracted Text
                   |
                   ↓
             Text Chunks
                   |
                   ↓
       Sentence Transformer
                   |
                   ↓
              Embeddings
                   |
                   ↓
                FAISS
             Vector Index
                   |
                   |
User Question ─────┘
       |
       ↓
Question Embedding
       |
       ↓
Retrieve Relevant Chunks
       |
       ↓
      Groq
       |
       ↓
    Answer
````

## 📁 Project Structure

```text
hr-policy-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── sample_hr_policy.txt
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/hr-policy-rag.git
```

### 2. Open the project folder

```bash
cd hr-policy-rag
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Groq API Key

You need a Groq API key to use the application.

Create your API key from the Groq developer platform.

When the Streamlit application starts, enter the API key in the
sidebar.

Do NOT upload your API key to GitHub.

## ▶️ Run the Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload an HR policy PDF and ask questions such as:

```text
How many annual leave days are employees entitled to?

How many days of sick leave are available?

What are the standard working hours?

Can employees work remotely?

What is the resignation notice period?
```

## 🎯 Example

Question:

```text
How many annual leave days do employees get?
```

Answer:

```text
Employees are entitled to 20 days of annual paid leave per year.
```

## 📚 What I Learned

Through this project, I learned the basic workflow of building a
RAG application:

* Reading documents with Python
* Text extraction
* Text chunking
* Embeddings
* Vector databases
* Similarity search
* Retrieval-Augmented Generation
* Working with LLM APIs
* Building applications with Streamlit
* Managing Python dependencies
* Using Git and GitHub

## 🔮 Future Improvements

Possible improvements include:

* Support multiple PDFs
* Add chat history
* Use a better text splitter
* Store FAISS indexes permanently
* Add source/page numbers
* Add document management
* Add authentication
* Use environment variables for API keys
* Deploy the application online

## 👩‍💻 Author

Faiza Kanwal

Built as a learning project to understand Retrieval-Augmented
Generation (RAG) and AI application development.

````

---

## How to put it on GitHub

Create a folder:

```text
hr-policy-rag
````

Put these 5 files inside:

```text
app.py
requirements.txt
README.md
sample_hr_policy.txt
.gitignore
```

Then open the terminal inside that folder:

```bash
git init
```

```bash
git add .
```

```bash
git commit -m "Build HR Policy RAG Assistant"
```

Then create a new repository on GitHub called:

```text
hr-policy-rag
```

Connect your local project:

```bash
git remote add origin https://github.com/YOUR-USERNAME/hr-policy-rag.git
```

Then:

```bash
git branch -M main
git push -u origin main
```

### ⚠️ One important security point

Don't do this:

```python
groq_api_key = "gsk_xxxxxxxxxxxxx"
```

and then push it to GitHub.

Your current beginner version asks for the key through Streamlit, so the key doesn't need to be stored in your code.

### The most important RAG concept to understand

For your learning, remember this simple idea:

**RAG = Retrieve first + Generate second**

The LLM does **not** need to know your HR policy beforehand.

For example:

```text
PDF
 ↓
"Employees get 20 days annual leave."
 ↓
FAISS retrieves that sentence
 ↓
Groq receives that sentence + user's question
 ↓
"Employees are entitled to 20 days..."
```

That's the core idea behind the application.
