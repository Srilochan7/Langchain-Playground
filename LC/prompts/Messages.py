import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms import Ollama

llm = Ollama(model="gemma:2b")

history = [
    SystemMessage(content="You are an ai assistant where you convert user inputs into pickup lines")
]

while(True):
    user_input = input("You :")

    # Check for exit condition IMMEDIATELY after getting input
    if user_input == "exit":
        break

    # If not exiting, then append to history and invoke LLM
    history.append(HumanMessage(content=user_input))
    result = llm.invoke(history)
    history.append(AIMessage(content=result))
    print(result)

# You might want to add a message after breaking the loop
print("Exiting chat. Goodbye!")