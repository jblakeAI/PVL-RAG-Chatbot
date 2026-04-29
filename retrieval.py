"""
retreival.py

Handles the full query -> answer pipeline:
1. Retrieves top- K chunks from Chroma vector database
2. Filters retrieved chunks using cross-encoder (keeps only relevant chunks)
3. Selects best clause if more than one clause passes the filter
4. Generates a grounded answer with LLM

Functions:

retrieval_dict -  retreives top-K chunks from Chroma vector database 

is_clause_relevant -  cross-encoder filter (returns only relevant clauses)

query_answer_pipe -  full end-to-end pipeline

"""


from typing import List, Dict
from langchain_chroma import Chroma
from config import (
    RETRIEVAL_K,
    CROSS_ENCODER_MODEL,
    RELEVANCE_THRESHOLD,
    MAX_RETRIEVAL_DIST
)

from sentence_transformers import CrossEncoder
from llm import llm_answer_generator, rewrite_query


_cross_encoder = None

### CHUNK RETRIEVAL ###

def retrieval_dict(db: Chroma, query: str, k: int = RETRIEVAL_K) -> List[Dict]:
    """
    Select the top k most similar chunks from Chroma for a given query.

    Returns a list of dicts with rank, L2 distance score, clause_id, and text.
    Note: Chroma returns L2 distance — LOWER score = BETTER match.
    
    """

    rows = []
    results = db.similarity_search_with_score(query, k=k) 

    for rank, (doc, score) in enumerate(results,1):
        rows.append({
            'rank': rank,
            'score': score,
            'clause_id':  doc.metadata.get('clause_id'),
            'text' : doc.page_content[:400]
            })
               
    return rows



### CROSS ENCODER ###

def get_cross_encoder():
    """
    Load cross encoder once 
    and reuse it.
    """
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(
            CROSS_ENCODER_MODEL, device="cpu",
              local_files_only=True
        )
    return _cross_encoder



###  CROSS ENCODER RELEVANCE FILTER ###

def is_clause_relevant(query: str, chunks: List[Dict], threshold = RELEVANCE_THRESHOLD) -> List[Dict]:   
    """ 
    Use the cross-encoder to score each chunk against the query.
    Returns only chunks that score at or above RELEVANCE_THRESHOLD.

    """
    encoder = get_cross_encoder()

    relevant_clauses = []
    for chunk in  chunks:
        clause_text = chunk['text']
        score = encoder.predict([(query, clause_text)])[0]
        if score >= threshold:
            chunk['cross_encoder_score'] = score
            relevant_clauses.append(chunk)

    return relevant_clauses




### PIPELINE ###

def query_answer_pipe(db, query):
    """
    End-to-end pipeline : user query -> response.
    Steps:
    1. Retrieve clause chunks
    2. Gate on retrieval distance (reject if no chunk is close enough)
    3. Select best clause if multiple pass the filter
    4. Generate answer with LLM
    
    Always returns a dict with three keys:
        answer      (str)       — the LLM response or a fallback message
        clause_id   (str|None)  — e.g. "3.02", or None if no clause was found
        clause_text (str|None)  — the raw clause text, or None if no clause was found
    """

    # Chunk retrieval
    chunks = retrieval_dict(db, query)

    if not chunks:
        return {
            "answer": "I could not find any relevant by-laws that match your question.",
            "clause_id": None,
            "clause_text": None
        } 
    
    
    # Numeric (distance) gate for retrieved clauses
    min_dist = min(chunks, key = lambda c: c['score'])   

    
    if min_dist['score'] > MAX_RETRIEVAL_DIST:                       
        return {
            "answer": "Your question doesn't appear to be covered by the current by-laws.",
            "clause_id": None,
            "clause_text": None
        }
    

    # Filtering answering clauses (Cross Encoder)                                
    candidates = is_clause_relevant(query, chunks)

    # --- QUERY REWRITE FALLBACK ---
    # If no clause passed the cross-encoder threshold, try once with a rewritten query.
    # The rewriter rephrases the user's natural language into legal terminology,
    # which can close the gap between casual phrasing and formal clause language.
    if not candidates:
        rewritten = rewrite_query(query)

        if rewritten != query:                              # Only retry if something actually changed
            chunks = retrieval_dict(db, rewritten)
            candidates = is_clause_relevant(rewritten, chunks)
            query = rewritten                               # Use rewritten query for LLM answer too

    if not candidates:
        return {
            "answer": "No clause was relevant enough to answer your question accurately.",
            "clause_id": None,
            "clause_text": None
        }
    # --- END FALLBACK ---


    if len(candidates) == 1:
        best_clause = candidates[0]
    else:
        best_clause = max(candidates, key=lambda c: c["cross_encoder_score"])


    # LLM response generation

    answer = llm_answer_generator(query, best_clause['text'])

    return {
        "answer": answer,
        "clause_id": best_clause['clause_id'],
        "clause_text": best_clause['text']
    }