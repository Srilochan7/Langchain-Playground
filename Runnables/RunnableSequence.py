from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence


model = Ollama(model="gemma:2b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="expain the following joke {joke}",
    input_variables=['joke']
)


chain = RunnableSequence(prompt1 ,model, prompt2, model, parser)

print(chain.invoke({'topic' : 'ai'}))