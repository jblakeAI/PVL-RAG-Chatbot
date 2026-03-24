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
from llm import llm_answer_generator



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

# Loaded once at module level so it isn't reloaded on every query.

cross_encoder = CrossEncoder(
   CROSS_ENCODER_MODEL,
    device="cpu"
)

###  CROSS ENCODER RELEVANCE FILTER ###

def is_clause_relevant(query: str, chunks: List[Dict], threshold = RELEVANCE_THRESHOLD) -> List[Dict]:   
    """ 
     Use the cross-encoder to score each chunk against the query.
    Returns only chunks that score at or above RELEVANCE_THRESHOLD.

    """

    relevant_clauses = []
    for chunk in  chunks:
        clause_text = chunk['text']
        score = cross_encoder.predict([(query, clause_text)])[0]
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
    
    """

    # Chunk retrieval
    chunks = retrieval_dict(db, query)

    if not chunks:
        return "I could not find any relevant by-laws that match your question."  
    
    
    # Numeric (distance) gate for retrieved clauses
    min_dist = min(chunks, key = lambda c: c['score'])         
    
    if min_dist['score'] > MAX_RETRIEVAL_DIST:                       
        return "Your question doesn't appear to be covered by the current by-laws."
    

    # Filtering answering clauses (Cross Encoder)                                
    candidates = is_clause_relevant(query, chunks)

    if not candidates:
        return "No clause was relevant enough to answer your question accurately."
    
    if len(candidates) == 1:
        best_clause = candidates[0]
    else:
        best_clause = max(candidates, key=lambda c: c["cross_encoder_score"])


    # LLM response generation

    answer = llm_answer_generator(query, best_clause['text'])

    return answer
    