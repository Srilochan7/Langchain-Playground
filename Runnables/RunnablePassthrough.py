from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough


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

pt = RunnablePassthrough()

jgc = RunnableSequence(prompt1, model, parser)

pc = RunnableParallel({
    'joke':RunnableSequence(prompt1, model, parser),
    'explanation': RunnableSequence(prompt2, model, parser)
})


fc = RunnableSequence(jgc, pc)

res = fc.invoke({'topic' : 'female rights'})


print(res)

