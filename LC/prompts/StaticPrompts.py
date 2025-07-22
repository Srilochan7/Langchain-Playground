import streamlit as st
from langchain_community.llms import Ollama

st.header("Research Paper Summarizer")

paper = st.text_input("Enter your input")

llm = Ollama(model="gemma:2b")


if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        prompt = f"Summarize this research paper: {paper}"
        result = llm.invoke(prompt)
        st.write(result.content if hasattr(result, "content") else result)
