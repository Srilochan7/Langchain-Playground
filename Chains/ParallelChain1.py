from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

model1 = Ollama(model="gemma:2b")
model2 = Ollama(model="gemma:2b")

prompt1 = PromptTemplate(
    template="Generate small and short notes from the following text \n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question&answers from the following text \n{topic}",
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': {"topic": RunnablePassthrough()} | prompt1 | model1 | parser,
    'quiz': {"topic": RunnablePassthrough()} | prompt2 | model2 | parser
})

merge_chain = {
    "notes": RunnablePassthrough(),
    "quiz": RunnablePassthrough()
} | prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Transformers are a groundbreaking neural network architecture introduced in 2017, revolutionizing Natural Language Processing (NLP) and expanding into computer vision and other domains. Unlike previous recurrent models (RNNs/LSTMs), they eschew sequential processing in favor of parallel computation, significantly speeding up training. Their core innovation is the "self-attention mechanism," which allows the model to weigh the importance of different parts of the input sequence to understand context. This mechanism generates "Query," "Key," and "Value" vectors for each token, calculating relevance scores between them. Multi-head attention is employed, allowing the model to focus on various aspects of the input simultaneously. Positional encodings are crucial, providing information about the order of tokens, as attention mechanisms alone are permutation-invariant. Transformers typically consist of an encoder and a decoder stack, each with multiple layers containing self-attention and feed-forward networks. They excel at capturing long-range dependencies, overcoming a major limitation of RNNs. Pre-trained transformer models like BERT and GPT have achieved state-of-the-art results across numerous NLP tasks, including machine translation, text generation, summarization, and question answering, and are foundational to modern large language models (LLMs).
"""

res = chain.invoke({'topic': text})

print(res)