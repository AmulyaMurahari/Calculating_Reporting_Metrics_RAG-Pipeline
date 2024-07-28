import streamlit as st
from PyPDF2 import PdfReader
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ''  # Ensure fallback for None text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_embeddings(text_chunks):
    embeddings_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    vectors = np.array([embeddings_model.encode(chunk) for chunk in text_chunks])
    return vectors

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
    index.add(embeddings.astype('float32'))  # Add embeddings to the index
    return index

def get_vector_store(text_chunks):
    embeddings = get_embeddings(text_chunks)
    return create_faiss_index(embeddings)

def enhanced_similarity_search(query, index):
    query_embedding = SentenceTransformer('paraphrase-MiniLM-L3-v2').encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), 1)  # Search for the most similar
    return indices.flatten()

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, 'answer is not available in the context', don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, index):
    indices = enhanced_similarity_search(user_question, index)
    docs = [text_chunks[i] for i in indices]  # Assuming text_chunks is globally accessible
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.title("💬 Chat with Your PDF Documents")
    st.subheader("Upload, Process, and Ask Questions")

    global text_chunks, faiss_index  # Declare globals to hold text chunks and the FAISS index

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Select PDF files", accept_multiple_files=True, type="pdf")
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    faiss_index = get_vector_store(text_chunks)
                    st.success("PDFs processed successfully")
                else:
                    st.warning("Please upload PDF files first")

    user_question = st.text_input("Ask a question based on the content of the uploaded PDFs:")
    
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Generating answer..."):
                user_input(user_question, faiss_index)
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()
