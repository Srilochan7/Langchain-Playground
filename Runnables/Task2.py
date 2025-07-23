from langchain_community.llms import Ollama 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableBranch, RunnablePassthrough

model = Ollama(model='gemma:2b')
parser = StrOutputParser()

# Step 1: Sentiment analysis
p1 = PromptTemplate(
    template="Determine if the text in one word is positive, negative, or neutral sentiment:\n{review}",
    input_variables=['review']
)

fc = (
    RunnableLambda(lambda d: d) |
    RunnableLambda(lambda d: {'text': d['text'], 'analyze': (p1 | model | parser).invoke({'review': d['text']})})
)

# Step 2: Word count
def wc(text): return len(text.split())

sc = RunnableLambda(lambda d: {**d, 'wc': wc(d['text'])})

# Step 3: Responses
p2 = PromptTemplate(
    template="Generate a short follow-up message for the following text:\n{text}",
    input_variables=['text']
)

p3 = PromptTemplate(
    template="Suggest a short, emphatic or helpful suggestion for this:\n{text}",
    input_variables=['text']
)

p4 = PromptTemplate(
    template="Provide a simple acknowledgment for this:\n{text}",
    input_variables=['text']
)

tc = RunnableBranch(
    (lambda d: "positive" in d['analyze'].lower(), p2 | model | parser),
    (lambda d: "negative" in d['analyze'].lower(), p4 | model | parser),
    (lambda d: "neutral" in d['analyze'].lower(), p3 | model | parser),
    RunnablePassthrough(),
)

# Final chain
final_chain = fc | sc | tc

# Test
print(final_chain.invoke({'text': 'very bad product'}))
