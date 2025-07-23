from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough


model = Ollama(model="gemma:2b")

def wc(text):
    return len(text.split())

parser = StrOutputParser()



prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)


jgc = RunnableSequence(prompt, model, parser)

pc = RunnableParallel({
    'joke':RunnablePassthrough(),
    'wc': RunnableLambda(wc)
})


fc = RunnableSequence(jgc, pc)

print(fc.invoke({'topic':'AI'}))

