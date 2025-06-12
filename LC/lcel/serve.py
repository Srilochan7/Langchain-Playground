from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)



gt = "Translate this following into {language} :"

gp = ChatPromptTemplate.from_messages(
    
        [("system", gt), ("user", "{text}")]
    
)

p = StrOutputParser()

chain = gt|model|p



app = FastAPI(title="Lanngchain server", version="1.0", description="A simple API server")


add_routes(
    app, 
    chain,
    path="/chain"
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)