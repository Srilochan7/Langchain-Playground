from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = Ollama(model="gemma:2b")

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser


res = chain.invoke({'topic' : 'cricket'})

print(res) 


chain.get_graph().print_ascii()
