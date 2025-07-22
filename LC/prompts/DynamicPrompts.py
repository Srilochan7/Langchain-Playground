import streamlit as st 
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

st.header("Reseach paper summariser")

llm = Ollama(model="gemma:2b")

paper = st.selectbox(
    "Select reseach paper :",
    ["Attention is all you need", "BERT: Pretraining of Bidirectional transformers", "GPT-3 Language models are Few-Short learners"]
)

style = st.selectbox(
    "Select yout style",
    ["Begineer-frienldy", "Deep dive", "Code oriented", "mathematical"]
)

length = st.selectbox(
    "Select your length",
    ["short", "medium", "detail"]
)

template = PromptTemplate(
    template = 
    """
    Read the following AI research paper and generate a summary that captures the core idea,
    methodology, key findings, and implications. Write it in a {style} tone and limit it to {length} words. 
    Here's the paper:
    {paper}
    """,
input_variables=["style", "length", "paper"]

)

prompt =  template.invoke(
    {
        'style':style,
        'length':length,
        'paper':paper
    }
)

if st.button("Summarise"):
    with st.spinner("Summarizing"):
        result = llm.invoke(prompt)
        
    st.write(result)


