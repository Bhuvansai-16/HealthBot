import os
import base64
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from mistralai import Mistral

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline

# -------------------------------
# ✅ 1️⃣ Streamlit page setup
# -------------------------------
st.set_page_config(page_title="📄✨ Smart Medical Chatbot", layout="centered")
st.title("📄✨ Smart Medical PDF + Prescription Chatbot (Gemma 3n)")

# -------------------------------
# ✅ 2️⃣ Load env vars
# -------------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not MISTRAL_API_KEY:
    st.error("❌ MISTRAL_API_KEY not set. Check your .env.")
    st.stop()

if not HUGGINGFACEHUB_API_TOKEN:
    st.error("❌ HUGGINGFACEHUB_API_TOKEN not set. Check your .env.")
    st.stop()

login(HUGGINGFACEHUB_API_TOKEN)
# -------------------------------
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
# -------------------------------
hf_llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3n-E4B-it",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.2,
    },
)

# -------------------------------
# ✅ 5️⃣ Sidebar uploads
# -------------------------------
with st.sidebar:
    st.header("📂 Upload Files")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF medical reports", type="pdf", accept_multiple_files=True
    )
    uploaded_image = st.file_uploader(
        "Upload handwritten prescription image", type=["png", "jpg", "jpeg"]
    )

# -------------------------------
# ✅ 6️⃣ Process PDFs
# -------------------------------
docs = []
if uploaded_pdfs:
    for uploaded_file in uploaded_pdfs:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

retriever = None
if chunks:
    st.sidebar.success(f"✅ {len(chunks)} PDF chunks ready.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
else:
    if uploaded_pdfs:
        st.sidebar.warning("⚠️ No valid text found in uploaded PDFs.")

# -------------------------------
# ✅ 7️⃣ Mistral OCR on prescription image
# -------------------------------
ocr_text = ""

def encode_image(uploaded_image):
    return base64.b64encode(uploaded_image.read()).decode("utf-8")

if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Prescription Image")
    st.sidebar.write("✅ Running Mistral OCR...")
    base64_image = encode_image(uploaded_image)

    ocr_response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        },
        include_image_base64=True
    )

    ocr_text = ocr_response.model_dump_json()
    st.sidebar.success("✅ Mistral OCR done (hidden)")

# -------------------------------
# ✅ 8️⃣ Chat memory & display
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------
# ✅ 9️⃣ Strict grounding prompt
# -------------------------------
grounded_template = """
You are a medical assistant.
Use ONLY the context below — do not use any outside knowledge.
If the context does not contain enough information to answer confidently,
say you do not know and recommend consulting a doctor.

Context:
{context}

Question:
{question}

Answer (clear, short, include precautions or treatments if possible, then remind the user to consult a doctor):
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=grounded_template,
)

# -------------------------------
# ✅ 🔟 Main chat input logic
# -------------------------------
if prompt := st.chat_input("Ask about your condition, reports, or prescription:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if retriever:
        retrieved_docs = retriever.get_relevant_documents(prompt)
        pdf_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        full_context = pdf_context

        if ocr_text:
            full_context += f"\n\nPrescription text:\n{ocr_text}"

        grounded_prompt = prompt_template.format(
            context=full_context,
            question=prompt
        )

        with st.spinner("💡 Gemma-3n is analyzing..."):
            answer = hf_llm.invoke(grounded_prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

        with st.expander("🔗 Used PDF Snippets"):
            for doc in retrieved_docs:
                st.write(doc.page_content[:300] + "...")

    elif ocr_text:
        only_context = f"Prescription text:\n{ocr_text}"
        grounded_prompt = prompt_template.format(
            context=only_context,
            question=prompt
        )

        with st.spinner("💡 Gemma-3n is analyzing prescription..."):
            answer = hf_llm.invoke(grounded_prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

    else:
        st.warning("❌ Please upload at least a PDF or prescription image to continue.")
