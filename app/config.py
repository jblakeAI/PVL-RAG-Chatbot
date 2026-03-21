''' 
config.py
Central configuration for the Pampellone Villas RAG chatbot.
Change paths and settings here and nowehre else.
 '''

from pathlib import Path



### PATHS ###

# If new documents paths are added update them here.

PROJECT_ROOT = Path(__file__).resolve().parent.parent # __file__ is config.py itself , parent.parent is the root folder 
PDF_PATHS = [PROJECT_ROOT / "data" / "Pampellone_by_laws.pdf"]   # Path designated as list for future additions of new bylaws or updated docs



### VECTOR DATABASE ###

# PERSIST_DIR  is where the Chroma saves the database                            
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME= "bylaws"




### EMBEDDING MODEL ###

# Free, local, no API key. Downloaded once from HuggingFace and cached.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"




### RETRIEVAL ### 

# The number of chunks to retreive from Chroma database before reranking 
RETRIEVAL_K = 3

# Maximum L2 distance score from Chroma to consider reranking 
# Lower MAX_RETREIVAL_DIST stricter filtering
MAX_RETREIVAL_DIST = 1.2




### CROSS ENCODER (RANKING MODEL) ###

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum cross-encoder score to consider a clause relevant
RELEVANCE_THRESHOLD = 2.0




### LLM (GROQ) ###

# Model avaialablity may change since deprecation of models occurs regularly
# Other models to use:
# llama-3.3-70b-versatile  — best quality, free tier, recommended
# llama-3.1-8b-instant     — faster, lighter, still very capable
# mixtral-8x7b-32768       — large context window

GROQ_MODEL = "llama-3.3-70b-versatile"

# Generation parameters
GROQ_MAX_TOKENS = 300    # Maximum tokens in the response.
GROQ_TIMEOUT = 30         # Seconds to wait for a response.                                                  !!!!! MAYBE ADD THIS


