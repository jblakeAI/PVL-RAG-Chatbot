"""
llm.py

Handles LLM answer generation via GROQ API.

Requires a new .env file in the project root with:
GROQ_API_KEY="your_key_here"

Functions:

llm_answer_generator  -   takes a query and a retrieved clause and returns a grounded answer string

"""


import os
from groq import Groq
from dotenv import load_dotenv

from config import GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_REWRITE_MODEL





### LOAD THE GROQ API KEY FROM THE .env FILE ###

load_dotenv()  
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)



### QUERY REWRITER ###

def rewrite_query(query: str) -> str:
    """
    Rewrite the user's query into formal legal language to improve
    cross-encoder matching against by-law clause text.

    Uses a smaller, faster model since this is a lightweight reformulation task.
    Called at most once per user query, only when the cross-encoder finds nothing.

    Args:
        query: the original user question

    Returns:
        A rewritten query string, or the original query if rewriting fails.
    """

    prompt = f"""Rewrite the following question using formal legal and property by-law terminology.
Keep it concise. Return ONLY the rewritten question, nothing else.

Original question: {query}
Rewritten question:"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_REWRITE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60          # Rewrites are short — cap tokens aggressively
        )
        rewritten = response.choices[0].message.content.strip()
       
        return rewritten
      
    except Exception:
        return query               # If rewriting fails, silently fall back to the original    




### ANSWER GENERATOR ###

def llm_answer_generator(query: str, clause: str) -> str:
    """
    
    Generate a grounded answer using the Groq API.
 
    The model is instructed to answer strictly from the provided clause —
    no speculation, no outside knowledge. This is what keeps the chatbot
    accurate and trustworthy for users.
 
    Args:
        query:  the user's question
        clause: the best matching clause text ranked from the vector DB
 
    Returns:
        Answer string from the LLM
    
    """

    prompt = f"""You are a legal analyst answering a question based strictly on the provided clause.

Rules:
- Only use the information provided in the clause.
- Do NOT speculate or use outside information.
- Always specify the clause ID number in the answer.
- If the clause does not answer the question sufficiently, say so explicitly.

Question:
{query}

Clause:
{clause}

Answer:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,               # Free, fast, accurate enough for legal Q&A
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,                # Keep at 0 — you want consistent, factual answers
        max_tokens=GROQ_MAX_TOKENS              
    )

    return response.choices[0].message.content.strip()