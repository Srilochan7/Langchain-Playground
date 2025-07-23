from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel


model = Ollama(model="gemma:2b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="generate a 1 line linkedin post on \n{topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate a 1 line X post on \n {topic}",
    input_variables=['topic']
)


parallel_chain = RunnableParallel({
    'linkedin' : RunnableSequence(prompt1, model, parser),
    'x' : RunnableSequence(prompt2, model, parser),
})

res = parallel_chain.invoke({'topic':'AI'})

print(res)
print(res['linkedin'])
print(res['x'])