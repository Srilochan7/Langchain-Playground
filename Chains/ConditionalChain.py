from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Model
model1 = Ollama(model="gemma:2b")

# Output schema
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give a sentiment of the feedback")

# Parser
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt
prompt1 = PromptTemplate.from_template(
    "Classify the sentiment of the following feedback into 'positive' or 'negative'.\n"
    "Feedback: {feedback}\n"
    "{format_instructions}"
)

# Chain
classifier_chain = prompt1.partial(format_instructions=parser2.get_format_instructions()) | model1 | parser2

# Inference
result = classifier_chain.invoke({'feedback': 'this is a wonderful smartphone'})
print(result.sentiment)
