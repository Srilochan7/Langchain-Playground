import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrival_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF uploads")
st.write("Upload Pdf's and chat with them")

api_key = st.text_input("Enter your Groq API key : ", type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It")
    
    session_id = st.text_input("Session ID", value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store={}
        
        
        uploaded_files = st.file_uploader("Choose a PDF file",type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            documents=[]
            for upload_file in uploaded_files:
                temppdf=f"./temp.pdf"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_files.getvalue())
                    file_name=uploaded_files.name
                    
                loader = PyPDFLoader