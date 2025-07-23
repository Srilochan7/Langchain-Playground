from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnableParallel, RunnableLambda, RunnablePassthrough


model = Ollama(model="gemma:2b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on : {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="summairize in 10 words the following text \n {joke}",
    input_variables=['joke']
)


rc = RunnableSequence(prompt1, model, parser)

cc = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough(),
)

fc = RunnableSequence(rc, cc)

print(fc.invoke({'topic':'Deeplearning'}))