from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = Ollama(model="gemma:2b")

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 3 pointer summary from the following text \n{topic}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic" : "unemployment in India"})

print(result)