import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from the .env file


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load content from content.txt
with open("content.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

# Chunking the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_text(document_text)

# Generate embeddings and store them in FAISS
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_texts(chunks, embeddings)

# OpenAI Model
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")

# Streamlit App
st.title("📖 RAG-based Q&A System")
st.write("Enter a question, and get an answer based on the document content.")

# Input box
question = st.text_input("Ask a question:", "")

if question:
    with st.spinner("Searching for an answer..."):
        # Search for relevant chunks
        relevant_docs = vector_db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Form the prompt
        prompt = f"Answer the following question using the given context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Get response from OpenAI
        response = chat_model.predict(prompt)

        # Display answer
        st.subheader("Answer:")
        st.write(response)
