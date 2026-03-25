"""
build.py

One- time script to ingest the by-laws PDF and build the Chroma vector database.

Run this ONCE from the terminal:
python build.py

After that, the database lives in the chroma_db/ folder and you never need
to run this again unless:
    - You add a new by-laws document
    - You change the PDF_PATHS in config.py
    - You delete or corrupt the chroma_db/ folder

"""
from ingestion import ingest_bylaws_pdfs
from config import PDF_PATHS
from config import PERSIST_DIR
from vectorstore import build_vectorstore


def main():
    print("=" * 50)
    print( "PVL by-laws Chatbot")
    print("=" * 50)

    print("\n [1/2] Ingesting PDF documnents...")
    chunks= ingest_bylaws_pdfs(PDF_PATHS)
    
    if not chunks:
        print("ERROR: No chunks found. Check PDF_PATHS in config.py.")
        return

    print(f"\n [2/2] Building Chroma vector database at '{PERSIST_DIR}'... ")

    db = build_vectorstore(chunks)

    print(f"\n Done! You may now run main.py to test the chatbot.")


if __name__ =="__main__":
    main()