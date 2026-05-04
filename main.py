"""
Pampellone By-Laws Chatbot - FastAPI Backend

This file only handles HTTP concerns: receiving requests, returning responses,
and error handling. All RAG logic is delegated to the existing project modules.
"""

import uvicorn
import google.auth
import gspread
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse

from langchain_chroma import Chroma
from vectorstore import load_db
from config import GROQ_MODEL, FEEDBACK_SHEET_ID
from retrieval import query_answer_pipe, get_cross_encoder

#### STARTUP ####

db: Chroma = None

@asynccontextmanager
async def lifespan(app: FastAPI):
   global db
   db = load_db()

   # FORCE model load at startup
   get_cross_encoder()

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
    allow_origins=["*"],
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
    clause_text: str | None = None    # The raw clause text (for transparency)


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    clause_id: str | None = None
    rating: str                       # "up" or "down"
    comment: str | None = None        # Only expected on thumbs down



#### ENDPOINTS ####

@app.get("/")                                   # Endpoint to serve the frontend HTML
async def serve_frontend():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "OK",  "message": "Pampellone By-Laws Chatbot is running."}

            
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400,  detail="Question cannot be empty.")
    
    try:
        result = query_answer_pipe(db, question)
        return AskResponse(
            answer=result["answer"],
            clause_id=result["clause_id"],
            clause_text=result["clause_text"]
        )

    except Exception as e:
        # This will show the real error in the chatbot UI
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Use the Cloud Run service account credentials automatically
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(FEEDBACK_SHEET_ID).sheet1

        sheet.append_row([
            datetime.now(timezone.utc).isoformat(),
            feedback.question,
            feedback.answer,
            feedback.clause_id or "",
            feedback.rating,
            feedback.comment or ""
        ])
        return {"status": "logged"}

    except Exception as e:
        # Never crash the app if feedback logging fails
        print(f"Feedback logging failed: {e}")
        raise HTTPException(status_code=500, detail="Feedback logging failed")


#### LOCAL ENTRY POINT ###

if __name__ == "__main__":
    uvicorn.run("main:app", host ="0.0.0.0", port=8080, reload=True)
