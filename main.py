"""
Pampellone By-Laws Chatbot - FastAPI Backend

This file only handles HTTP concerns: receiving requests, returning responses,
and error handling. All RAG logic is delegated to the existing project modules.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain_chroma import Chroma
from vectorstore import load_db
from config import GROQ_MODEL
from retrieval import query_answer_pipe



#### STARTUP ####

db: Chroma = None

@asynccontextmanager
async def lifespan(app: FastAPI):
   global db
   db = load_db()
   print(f"Server ready - model: {GROQ_MODEL}")
   yield 




#### APP SETUP ####

app = FastAPI(
    title = "Pampellone By-laws Chatbot",
    description = "Answers questions about the Pampellone Villas by-laws.",
    version = "1.0.0",
    lifespan=lifespan
)


## CORS : frontend <--> server communication ##

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://huggingface.co/spaces/j-blake/PVL_RAG_Chatbot"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)



#### REQUEST AND RESPONSE MODELS ####

# Pydantic models describe the exact shape of JSON that comes IN and goes OUT
# FastAPI validates these automatically 

class AskRequest(BaseModel):
    question: str            # The user's question (required)
        

class AskResponse(BaseModel):
    answer: str                       # GROQ's grounded answer
    clause_id: str | None = None      # The clause that answered the question (if found)
    clause_text: str | None = None    # # The raw clause text (for transparency)





#### ENDPOINTS ####
@app.get("/")
async def health_check():
    return {"status": "OK",  "message": "Pampellone By-Laws Chatbot is running."}

            
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400,  detail="Question cannot be empty.")
    
    result = query_answer_pipe(db, question)

    return AskResponse(
        answer=result["answer"],
        clause_id=result["clause_id"],
        clause_text=result["clause_text"]
    )



    #### LOCAL ENTRY POINT ###

if __name__ == "__main__":
    uvicorn.run("main:app", host ="0.0.0.0", port=8000, reload=True)