import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"  # corrected typo in variable name

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title('Langchain Demo with Ollama')
input_text = st.text_input("What is the question you have in mind?")

# Set up LLM
llm = Ollama(model="gemma:2b")  # keyword argument required for model name

# Output parser
output_parser = StrOutputParser()

# Create the chain
chain = prompt | llm | output_parser

# Handle user input
if input_text:
    result = chain.invoke({"question": input_text})
    st.write(result)
