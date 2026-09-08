```python
import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# 1. PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 HR Policy Assistant")

st.write(
    "Upload an HR policy PDF and ask questions about the policy."
)


# ==========================================
# 2. GROQ API KEY
# ==========================================

try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)

except Exception:
    st.error(
        "Groq API key not found. "
        "Please add GROQ_API_KEY in Streamlit Secrets."
    )
    st.stop()


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


model = load_embedding_model()


# ==========================================
# 4. READ PDF
# ==========================================

def extract_text_from_pdf(pdf_file):

    pdf = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in pdf:

        page_text = page.get_text()

        text += page_text + "\n"

    pdf.close()

    return text


# ==========================================
# 5. SPLIT TEXT INTO CHUNKS
# ==========================================

def create_chunks(text):

    words = text.split()

    chunks = []

    chunk_size = 300

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():

            chunks.append(chunk)

    return chunks


# ==========================================
# 6. CREATE FAISS INDEX
# ==========================================

def create_faiss_index(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ==========================================
# 7. SEARCH RELEVANT CHUNKS
# ==========================================

def search_documents(
    question,
    chunks,
    index,
    top_k=3
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    distances, results = index.search(
        question_embedding,
        top_k
    )

    relevant_chunks = []

    for index_number in results[0]:

        if index_number < len(chunks):

            relevant_chunks.append(
                chunks[index_number]
            )

    return relevant_chunks


# ==========================================
# 8. ASK GROQ
# ==========================================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question using ONLY the
information provided in the HR policy context.

Do not use outside knowledge.

Do not make up information.

If the answer is not available in the
provided policy, say:

"I could not find this information in
the uploaded HR policy."

Keep the answer clear and concise.

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


# ==========================================
# 9. UPLOAD PDF
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your HR Policy PDF",
    type=["pdf"]
)


# ==========================================
# 10. PROCESS PDF
# ==========================================

if uploaded_file:

    with st.spinner(
        "Reading your HR policy..."
    ):

        text = extract_text_from_pdf(
            uploaded_file
        )

    if not text.strip():

        st.error(
            "No readable text was found "
            "in this PDF."
        )

        st.stop()


    st.success(
        "PDF uploaded successfully!"
    )


    # Create chunks

    chunks = create_chunks(text)

    st.info(
        f"Created {len(chunks)} text chunks."
    )


    # Create FAISS index

    with st.spinner(
        "Creating document search index..."
    ):

        index = create_faiss_index(
            chunks
        )


    st.success(
        "HR policy is ready! You can ask questions."
    )


    # ==========================================
    # 11. QUESTION
    # ==========================================

    question = st.text_input(
        "💬 Ask a question about the HR policy"
    )


    if question:

        with st.spinner(
            "Searching the HR policy..."
        ):

            relevant_chunks = search_documents(
                question,
                chunks,
                index,
                top_k=3
            )


        # Combine retrieved chunks

        context = "\n\n".join(
            relevant_chunks
        )


        # Generate answer

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                context
            )


        # ==========================================
        # 12. DISPLAY ANSWER
        # ==========================================

        st.subheader("💬 Answer")

        st.write(answer)


        # ==========================================
        # 13. SHOW RETRIEVED INFORMATION
        # ==========================================

        with st.expander(
            "🔎 View retrieved policy sections"
        ):

            for number, chunk in enumerate(
                relevant_chunks,
                start=1
            ):

                st.markdown(
                    f"**Policy Section {number}**"
                )

                st.write(chunk)

                st.divider()
```

### Your Streamlit Secret must be exactly:

```toml
GROQ_API_KEY = "GROQ_API_KEY"
```

And **do not put the actual key in GitHub**.

After replacing `app.py`:

```bash
git add app.py
git commit -m "Build HR Policy RAG assistant"
git push
```

.
