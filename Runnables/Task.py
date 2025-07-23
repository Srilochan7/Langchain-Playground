from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

model = Ollama(model="gemma:2b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="analyze this following statement and determine if it is a joke or a question. Respond with 'joke' or 'question'.\nStatement: {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Summarize the following question:\nQuestion: {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Rate the humor of the following joke on a scale of 1 to 5:\nJoke: {text}",
    input_variables=['text']
)

def wc(text):
    return len(text.split())

pc = RunnableParallel({
    'analyze': RunnableSequence(prompt1 | model | parser),
    'length': RunnableLambda(lambda x: wc(x['text'])), # <--- CORRECTED LINE HERE
    'text': RunnablePassthrough()
})

cc = RunnableBranch(
    (lambda d: "joke" in d['analyze'].lower(), prompt3 | model | parser),
    (lambda d: "question" in d['analyze'].lower(), prompt2 | model | parser),
    RunnableLambda(lambda d: f"Unclassified input. Analysis result: {d['analyze']}")
)

full_chain = pc | cc

print(full_chain.invoke({'text': 'tell me a joke about AI'}))
print(full_chain.invoke({'text': 'What is the capital of France?'}))
print(full_chain.invoke({'text': 'This is a random statement.'}))