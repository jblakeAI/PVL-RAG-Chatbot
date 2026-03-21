"""
vectorstore.py

Handles creating and loading the Chroma vector database.

Functions:

init_embeddings   -   loads the sentence-transformer embedding model

build_vectorstore -   creates the vectorstore db from chunks (RUN ONCE)

load_db           -   loads existing database 
"""




# CREATE EMBEDDINGS AND VECTORSTORE 
import os
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


from config import(
  PERSIST_DIR, 
  COLLECTION_NAME,
  EMBEDDING_MODEL 
)




### LOAD EMBEDDINGS MODEL ###

def init_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformer embedding model.
    all-MiniLM-L6-v2 runs on CPU and is fast enough for this use case.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )




### BUILD VECTORSTORE ### 

def build_vectorstore(chunks: List[Dict]) -> Chroma:
    '''
    Create a Chroma database from clause chunks and saves it to disk.

    This only needs to be run ONCE or when source documents are updated or changed.
    After that, use load_db() instead. 


    Args:
        chunks: output from ingest_bylaws_pdfs() — list of {metadata, text} dicts
 
    Returns:
        Chroma db instance

    '''

    embeddings = init_embeddings()
                                 
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]

    db = Chroma.from_texts(         
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    
    )
    print(f"Vector DB persisted to: {PERSIST_DIR}")
    print(f"Total vectors stored: {db._collection.count()}")

    return db



### LOAD DATABASE ###

def load_db() -> Chroma:
    """
    Load an existing Chroma database from disk.
    Run build_vectorstore() first if the DB doesn't exist yet.
 
    Returns:
    Chroma db instance ready for similarity search
   
    """
    assert os.path.exists(PERSIST_DIR), (
        f"Vector DB not found at '{PERSIST_DIR}'." 
        "Run build.py first to create it."
        )
    
    embeddings = init_embeddings()

    db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
    count = db._collection.count
    assert count() > 0, (
        f"DB loaded but contains no vectors. Rerun build.py"
    )
    
    print(f"Vector DB load from {PERSIST_DIR} and the count is: {count} ")
    return db